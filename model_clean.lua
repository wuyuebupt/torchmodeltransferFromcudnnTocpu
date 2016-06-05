--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  clean up a model before saving
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

-- reference
-- https://github.com/torch/DEPRECEATED-torch7-distro/issues/47
function zeroDataSize(data)
	if type(data) == 'table' then
		for i = 1, #data do
			data[i] = zeroDataSize(data[i])
		end
	elseif type(data) == 'userdata' then
		data = torch.Tensor():typeAs(data)
	end
	return data
end

-- Resize the output, gradInput, etc temporary tensors to zero (so that the on disk size is smaller)
function cleanupModel(node)
	if node.output ~= nil then
		node.output = zeroDataSize(node.output)
	end
	if node.gradInput ~= nil then
		node.gradInput = zeroDataSize(node.gradInput)
	end
	if node.finput ~= nil then
		node.finput = zeroDataSize(node.finput)
	end
	-- Recurse on nodes with 'modules'
	if (node.modules ~= nil) then
		if (type(node.modules) == 'table') then
			for i = 1, #node.modules do
				local child = node.modules[i]
				cleanupModel(child)
			end
		end
	end
	collectgarbage()
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
local softMaxLayer = cudnn.SoftMax():cuda()
-- add Softmax layer
model:add(softMaxLayer)
-- Evaluate mode
model:evaluate()

print(model)
cleanupModel(model)
torch.save(arg[2], model)

