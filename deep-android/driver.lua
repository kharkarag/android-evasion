-- Example of how to test using a pre-trained network
-- Expects a directory containing two or more directories
-- One directory contains all the malware
-- The other directory contains all the benign software

-- given a model that has already been trained
-- and a directory containing programs - classify into malware / benign

require 'nn'
require 'optim'
require 'nngraph'
require 'cunn'
require 'cutorch'

require 'readMalwareData'
require 'testModel'

cmd = torch.CmdLine()
cmd:option('-useCUDA',false,'use CUDA optimisation')
cmd:option('-dataDir','./eval/','directory with the android programs to classify')
cmd:option('-modelPath','./trainedNets/model.th7','path to model to use for testing')
opt = cmd:parse(arg)

--print('loading model from disk')
savedModel = torch.load(opt.modelPath)
--print('loaded model')
--print(savedModel.trainedModel)

-- we need these values to correctly prepare the files when reading from disk
opt.programLen = savedModel.opt.programLen
opt.kernelLength = savedModel.opt.kernelLength
opt.maxSequenceLength = savedModel.opt.maxSequenceLength

--print('reading data from disk')
allData = readMalwareData(opt.dataDir,savedModel.metaData)

if opt.useCUDA then
	savedModel.trainedModel:cuda()
end
savedModel.trainedModel:evaluate()

--print('starting test')
scores, testResult,confmat,time = testModel(allData,savedModel.trainedModel,0)

--[[
print('Results')
print('f-score   ',testResult.fscore)
print('precision ',testResult.prec)
print('recall    ',testResult.recall)
print('accuracy  ',testResult.accuracy)
print('--')
print('Confusion Matrix')
print(confmat)
print('--')
print('time to complete test (s) :',time)
--]]

--print(allData.program)

print(scores)
