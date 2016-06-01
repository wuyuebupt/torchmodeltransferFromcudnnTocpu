--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'


function replaceModules(net, orig_class_name, replacer)
	local nodes, container_nodes = net:findModules(orig_class_name)
	print(nodes)
	for i = 1, #nodes do
		for j = 1, #(container_nodes[i].modules) do
			if container_nodes[i].modules[j] == nodes[i] then
				local orig_mod = container_nodes[i].modules[j]
				container_nodes[i].modules[j] = replacer(orig_mod)
			end
		end
	end
end

function cudnnNetToCpu(net)
	-- this line will double the memory
	-- local net_cpu = net:clone():float()
	-- this line not
	local net_cpu = net:float()
	print(net_cpu)
	
	replaceModules(net_cpu, 'cudnn.SpatialConvolution', 
	function(orig_mod)
		local cpu_mod = nn.SpatialConvolutionMM(orig_mod.nInputPlane, orig_mod.nOutputPlane,
		orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH, orig_mod.padW, orig_mod.padH)
		cpu_mod.weight:copy(orig_mod.weight)
		print(orig_mod)
		if orig_mod.bias then
			cpu_mod.bias:copy(orig_mod.bias)
		end
		return cpu_mod
	end)
	
	replaceModules(net_cpu, 'cudnn.ReLU', function() return nn.ReLU() end)
	replaceModules(net_cpu, 'cudnn.SpatialBatchNormalization', function() return nn.SpatialBatchNormalization() end)
	replaceModules(net_cpu, 'cudnn.SpatialAveragePooling', function(orig_mod) local cpu_mod =  nn.SpatialAveragePooling(
		orig_mod.kW, orig_mod.kH, orig_mod.dW, orig_mod.dH)
		return cpu_mod end)
	return net_cpu
end


if #arg ~= 2 then
   io.stderr:write('Usage: th model_transfer.lua [MODELIN] [MODELOUT]\n')
   os.exit(1)
end
-- for _, f in ipairs(arg) do
f = arg[1]
if not paths.filep(f) then
   io.stderr:write('file not found: ' .. f .. '\n')
   os.exit(1)
end

-- Load the model
local model = torch.load(arg[1])
-- local softMaxLayer = cudnn.SoftMax():cuda()
-- add Softmax layer
-- model:add(softMaxLayer)
model = cudnnNetToCpu(model)
local softMaxLayer = nn.SoftMax()
model:add(softMaxLayer)
-- Evaluate mode
model:evaluate()
model = model:float()
print(model)
torch.save(arg[2], model)


