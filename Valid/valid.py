import numpy as np
from tvm import relay
import topi
import tvm
from tvm.contrib import graph_runtime

def Init(data, shape, mod=False):
    datum=int(1)
    for OC in range(0,shape[0]):
        for IC in range(0,shape[1]):
            for H in range(0,shape[2]):
                for W in range(0,shape[3]):
                    data[OC][IC][H][W] = float(datum)
                    datum = datum % int(24)
                    datum += 1

    return

P=2
S=3

strides = (S,S)
#####################
padding = (P,P,P,P)
#####################
dilation = (1,1)

data_shape = (1,32,112,112)
w_shape = (64,32,3,3)


guess = data_shape[3]*padding[0] + padding[0]
data = relay.var(name_hint="data", shape=data_shape)
weight = relay.var(name_hint='weight', shape=w_shape)
P = weight.type_annotation.concrete_shape


conv2d = relay.nn.conv2d(data=data,weight=weight,strides=strides, padding=padding, channels=P[0],kernel_size=(P[2],P[3]), groups=1 )

function = relay.Function( [data,weight] , conv2d ) 
model = tvm.IRModule.from_expr(function)


input_data = np.zeros(data_shape).astype("float32")
input_weight = np.zeros(w_shape).astype("float32")


##############################
Init(input_data,data_shape)
Init(input_weight,w_shape)

################################
exe_p = {weight.name_hint:tvm.nd.array(input_weight)}

with tvm.transform.PassContext(opt_level=3):
    graph, lib, PARAMS = relay.build(model, 'llvm', params=exe_p)

ctx = tvm.runtime.cpu()
module = graph_runtime.create(graph, lib, ctx)

module.set_input("data",input_data)
for p in PARAMS:
    module.set_input(str(p),PARAMS[p])

module.run()
output_data = module.get_output(0).asnumpy().astype("float32")
'''
print((output_data.shape))
output_data.astype("float32").tofile("../valid.bin")
'''

#np.save("./delta.txt",output_data)
input_data = tvm.te.placeholder(shape=data_shape, name="input_data")
p1 = tvm.te.placeholder(shape=w_shape,name="parameter" )

conv = topi.nn.conv2d(input=input_data, filter=p1,strides=strides, padding=padding, dilation=dilation, groups=1)

with tvm.target.create('llvm'):
    sch = topi.generic.schedule_conv2d_nchw([conv])


print( tvm.lower(sch, [input_data,p1], simple_mode=True) )




