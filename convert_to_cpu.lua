--[[
A quick script for converting GPU checkpoints to CPU checkpoints.
CPU checkpoints are not saved by the training script automatically
because of Torch cloning limitations. In particular, it is not
possible to clone a GPU model on CPU, something like :clone():float()
with a single call, without needing extra memory on the GPU. If this
existed then it would be possible to do this inside the training
script without worrying about blowing up the memory.
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn' -- only needed if the loaded model used cudnn as backend. otherwise can be commented out
-- local imports

-- cmd = torch.CmdLine()
-- cmd:text()
-- cmd:text('Convert a GPU checkpoint to CPU checkpoint.')
-- cmd:text()
-- cmd:text('Options')
-- cmd:argument('-model','GPU model checkpoint to convert')
-- cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
-- cmd:text()

model = '/media/amds/data2/dataset/resnet/resnext_101_64x4d.t7'
savefile = '/media/amds/data2/dataset/resnet/resnext_101_64x4d_cpu.t7'
gpuid = 0

-- parse input params
-- local opt = cmd:parse(arg)
torch.manualSeed(123)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
cutorch.setDevice(gpuid + 1) -- note +1 because lua is 1-indexed

local checkpoint = torch.load(model)
print(checkpoint)
-- local protos = checkpoint.protos

-------------------------------------------------------------------------------
-- these functions are adapted from Michael Partheil
-- https://groups.google.com/forum/#!topic/torch7/i8sJYlgQPeA
-- the problem is that you can't call :float() on cudnn module, it won't convert
function replaceModules(net, orig_class_name, replacer)
    local nodes, container_nodes = net:findModules(orig_class_name)
    for i = 1, #nodes do
        for j = 1, #(container_nodes[i].modules) do
            if container_nodes[i].modules[j] == nodes[i] then
                local orig_mod = container_nodes[i].modules[j]
                print('=================start==================')
                print('replacing a cudnn module with nn equivalent...')
                print(orig_mod)
                container_nodes[i].modules[j] = replacer(orig_mod)
                print('=================end==================')
            end
        end
    end
end

function cudnnNetToCpu(net)
    local net_cpu = net:clone():float()
    replaceModules(net_cpu, 'cudnn.SpatialConvolution', function(orig_mod)
        local cpu_mod = nn.SpatialConvolution(orig_mod.nInputPlane, orig_mod.nOutputPlane,
          orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
        print(orig_mod)
        print(cpu_mod)
        cpu_mod.weight:copy(orig_mod.weight)
        if orig_mod.bias then
            cpu_mod.bias:copy(orig_mod.bias)
        else
            cpu_mod.bias = nil
        end
        cpu_mod.gradWeight = nil -- sanitize for thinner checkpoint
        cpu_mod.gradBias = nil -- sanitize for thinner checkpoint
        return cpu_mod
    end)
    replaceModules(net_cpu, 'cudnn.SpatialMaxPooling', function(orig_mod)
        local cpu_mod = nn.SpatialMaxPooling(orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH,
                                           orig_mod.padW, orig_mod.padH)
        return cpu_mod
    end)
    replaceModules(net_cpu, 'cudnn.SpatialAveragePooling', function(orig_mod)
        local cpu_mod = nn.SpatialAveragePooling(orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH,
                                           orig_mod.padW, orig_mod.padH)
        return cpu_mod
    end)
    replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
    return net_cpu
end
-------------------------------------------------------------------------------

local cpu_cnn = cudnnNetToCpu(checkpoint)

torch.save(savefile,cpu_cnn)


