import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import pickle
import numpy as np
import math
import tvm
from tvm.contrib.download import download_testdata
alpha_root = "/home/alpha930/Desktop/CNetProj/binary_data/"
test_root = "/home/alpha930/Desktop/CNetProj/Param/"
root = '/home/alpha930/Desktop/CNetProj/Weights/'

def Validate(t1,t2,bound=0.00001):
    if np.all( np.abs(t1-t2)< bound ):
        return True
    else:
        return False

def saveDataToBin(name,data):
    if isinstance(data, np.ndarray):
        data.astype("float32").tofile( alpha_root + str(name)+".bin")
    else:
        data.data.numpy().astype("float32").tofile( alpha_root + str(name)+".bin")
    return

def numpyToTensor(data, Oshape=None):
    temp =torch.from_numpy(np.array(data))
    return temp

def tensorToNumpy(data):
    temp = data.data.numpy().astype("float32")
    return temp

def WhoAreYou(layer):
    layer_tag = "None"
    if isinstance( layer, torch.nn.Conv2d):
        layer_tag="Conv"
    elif isinstance( layer ,torch.nn.BatchNorm2d):
        layer_tag="BatchNorm"
    elif isinstance( layer, torch.nn.ReLU6):
        layer_tag="Relu6"
    
    return layer_tag
# /home/alpha930/Desktop/CNetProj/Weights/layer_0_ConvBNRelu
def ExetractParams(layer, b_i, block_name ,l_index):
    layer_tag = WhoAreYou(layer)
    name = root+"layer_"+str(b_i)+"_"+block_name+"/"

    if layer_tag == "None":
        print("Fail to Extract Tag")
    else:
        if layer_tag == "Conv":
            target_path = name+str(l_index)+"_"+layer_tag+".bin"
            data = layer.weight.data
            #data.data.numpy().astype("float32").tofile(target_path)
            print(data.shape)

        elif layer_tag == "BatchNorm":
            rm = layer.running_mean
            rv = layer.running_var
            gamma = layer.weight.data
            beta = layer.bias.data
            print(rm.shape)

            '''
            target_path = name+str(l_index)+"_"+layer_tag+"_mean"+".bin"            
            rm.data.numpy().astype("float32").tofile(target_path)
            target_path = name+str(l_index)+"_"+layer_tag+"_var"+".bin"
            rv.data.numpy().astype("float32").tofile(target_path)
            target_path = name+str(l_index)+"_"+layer_tag+"_gamma"+".bin"
            gamma.data.numpy().astype("float32").tofile(target_path)
            target_path = name+str(l_index)+"_"+layer_tag+"_beta"+".bin"
            beta.data.numpy().astype("float32").tofile(target_path)
            '''

def SimpleTest():
    G=32
    input_data = np.random.uniform(-1,1,size=(1,32,112,112)).astype('float32')
    kernel = np.random.uniform(-1,1,size=(64,32//G,3,3)).astype('float32')
    
    '''
    input_data = []
    for i in range(0,1*4*5*5):
        input_data.append(i)

    kernel = []
    for i in range(0,4*2*3*3):
        kernel.append(i)
    '''
    input_data = np.array(input_data)
    kernel = np.array(kernel)

    #input_data = np.reshape(input_data,(1,32,112,112))
    #kernel = np.reshape(kernel,(64,32,3,3))

    i = torch.from_numpy(np.array(input_data))
    k = torch.from_numpy(np.array(kernel))
    conv = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3), stride=(1,1), padding=(0,0),groups = G, bias=False)
    conv.weight.data = k

    model = conv.eval()## do Inference Mode
   
    o = model(i)

    output = tensorToNumpy(o)
    test_i = tensorToNumpy(i)
    test_k = tensorToNumpy(k)
    saveDataToBin("input",test_i)
    saveDataToBin("filter",test_k)
    saveDataToBin("output",output)

    print(output.shape)
    print(output)
    return


def isConvBNRelu(block):
    if isinstance(block , torchvision.models.mobilenet.ConvBNReLU):
        return True
    return False

def isInvertedResidual(block):
    if isinstance(block , torchvision.models.mobilenet.InvertedResidual):
        return True
    return False

def GetMVParam(get_out=False):
    mv2 = models.mobilenet_v2(pretrained=True)
    mv2 = mv2.eval()
    mv2_tag = mv2.features
    #input_data = np.random.uniform(-1,1,size=(1,3,224,224))
    #input_data.astype("float32").tofile("/home/alpha930/Desktop/CNetProj/Weights/input/input_data.bin")
    #np.save("/home/alpha930/Desktop/CNetProj/python/input_data.bin",input_data)
    #input_data = np.load("/home/alpha930/Desktop/CNetProj/python/input_data.npy")

    for block_index, block in enumerate(mv2_tag):
        layer_index = 0
        block_name=""
        if isConvBNRelu(block):
            print("######ConvBNRelu######")
            block_name = "ConvBNRelu"
            for layer in block:
                ExetractParams(layer,block_index, block_name,layer_index)
                layer_index += 1
        elif isInvertedResidual(block):
            print("######InvertedResidual######")
            block_name = "InvertedResidual"
            for layer in block.conv:
                if isConvBNRelu(layer):
                    for layer_sub in layer:
                        ExetractParams(layer_sub,block_index, block_name, layer_index)
                        layer_index +=1
                else:
                    ExetractParams(layer,block_index, block_name, layer_index)
                    layer_index +=1
        print("")
    mv2_tag = mv2.classifier
    print("######classifier######")
    for block in mv2_tag:
        print(block)
    return 

def GetParam(get_out=False):
    mv2 = models.mobilenet_v2(pretrained=True)
    mv2 = mv2.eval()

    mv2_config = [
            'ConvBNRelu' #0
            , 'InvertedResidual' #1
            , 'InvertedResidual' #2
            , 'InvertedResidual' #3
            , 'InvertedResidual' #4
            , 'InvertedResidual' #5
            , 'InvertedResidual' #6
            , 'InvertedResidual' #7
            , 'InvertedResidual' #8
            , 'InvertedResidual' #9
            , 'InvertedResidual' #10
            , 'InvertedResidual' #11
            , 'InvertedResidual' #12
            , 'InvertedResidual' #13
            , 'InvertedResidual' #14
            , 'InvertedResidual' #15
            , 'InvertedResidual' #16
            , 'InvertedResidual' #17
            , 'InvertedResidual' #18
            , 'Softmax' # classifier
            ]

    data = []
    for layer_index, name, param in enumerate( mv2.named_parameters() ):
        data.append({"name":name, 'data':param.data.numpy()})
        param.data.numpy().astype("float32").tofile(test_root+str(name)+".bin")
    
    if get_out:
        with open("./mobilenet_v2.pkl", 'wb') as f:
            pickle.dump(data,f)
        with open("./mobilenet_v2.pkl","rb") as f:
            load_data = pickle.load(f)
    return 

def cutLayer(start_p, end_p, save_flag=False):
    data = []
    mv2_cut = models.mobilenet_v2(pretrained=True).features[start_p:end_p]
    mv2_cut = mv2_cut.eval()

    ### IC = 3


    return


def BASELINE():
    import torch
    mv2 = models.mobilenet_v2(pretrained=True)
    mv2 = mv2.eval()

    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.eval()

    from PIL import Image
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))

    # Preprocess the image and convert to tensor
    from torchvision import transforms

    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)

    synset_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_synsets.txt",
        ]
    )
    synset_name = "imagenet_synsets.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(" ") for line in synsets]
    key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

    class_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_classes.txt",
        ]
    )
    class_name = "imagenet_classes.txt"
    class_path = download_testdata(class_url, class_name, module="data")
    with open(class_path) as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    # Convert input to PyTorch variable and get PyTorch result for comparison
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        output = model(torch_img)
        output2 = mv2(torch_img)
        # Get top-1 result for PyTorch
        top1_torch = np.argmax(output.numpy())
        torch_class_key = class_id_to_key[top1_torch]

    print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))

    return


def MVTest():
    input_data = np.load("/home/alpha930/Desktop/CNetProj/python/input_data.npy").astype("float32")
    input_data = numpyToTensor(input_data)

    s_pos=0
    e_pos =0
    for i in range(19):
        #name = root+"layer_"+str(i)+"_"+block_name+"/"
        if i ==0 or i==18:
            target_path = root+"layer_"+str(i)+"_"+"ConvBNRelu"+"/"
        else:
            target_path = root+"layer_"+str(i)+"_"+"InvertedResidual"+"/"
        print(target_path)
        e_pos=i+1
        layer = models.mobilenet_v2(pretrained=True).features[s_pos:e_pos]
        layer.eval()
        input_data = layer(input_data)
        input_data.data.numpy().astype("float32").tofile(target_path+"imm_out.bin")

        s_pos = e_pos

    return

def Test():
    input_data = np.load("/home/alpha930/Desktop/CNetProj/python/input_data.npy").astype(np.float32)
    input_data = numpyToTensor(input_data)

    layer = models.mobilenet_v2(pretrained=True).features[0:4]
    layer.eval()

    out_original = layer(input_data)
    #out = out_original


    layer = models.mobilenet_v2(pretrained=True).features[0:4]
    layer.eval()

    out_test_mid = layer[0:3](input_data)
    L = layer[3](out_test_mid)
    if Validate( tensorToNumpy(out_original), tensorToNumpy(L)):
        print("Done?")


    print(L)
    #out_test = layer[3](out_test_mid)
    out_test = L.conv[0](out_test_mid)
    out_test = L.conv[1](out_test)
    out_test = L.conv[2](out_test)
    out_test = L.conv[3](out_test)
    
    print(L.conv[0])
    print(L.conv[1])
    print(L.conv[2])
    print(L.conv[3])

    if Validate( tensorToNumpy(out_original), tensorToNumpy(out_test)):
        print("Done?")


    #out.data.numpy().astype("float32").tofile(alpha_root+"output.bin")
    #print(out.shape)
    #print(output)
    return

'''
[[[ 0.01540584 -0.04439151  0.0722454 ]
  [-0.00241359 -0.1428622   0.7564706 ]
  [ 0.03497745  0.10049933 -0.7938975 ]]]
'''
if __name__ =="__main__":
    print("Start Testing")
    print("")
    #BASELINE()
    #cutLayer(0,1,save_flag=True)
    #SimpleTest()
    #GetMVParam()
    #MVTest()
    Test()



