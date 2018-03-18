require 'nn'
require 'optim'
require 'nngraph'

require 'readMalwareData'
require 'splitMalwareData'
require 'buildNetwork'
require 'trainModel'

local cmd = torch.CmdLine()
cmd:option('-seed',1,'seed the random number generator')
cmd:option('-nEmbeddingDims',8,'number of dims in lookupTable for projecting instructions to network')
cmd:option('-nConvFilters',64,'number of convolutional filters')
cmd:option('-kernelLength',8,'seed the random number generator')
cmd:option('-useHiddenLayer',true,'use hidden layer between the conv layers and classifier')
cmd:option('-nHiddenNodes',16,'seed the random number generator')
cmd:option('-weightClasses',false,'seed the random number generator')
cmd:option('-nSamplingEpochs',10,'how often to sample the validation set - slow')
cmd:option('-useDropout',false,'use dropout between the conv and hidden layers')
cmd:option('-dropoutFrac',0.5,'dropout strength')
cmd:option('-randomize',false,'randomly select the network parameters')
cmd:option('-numDAShuffles',1,'number of function order shuffled versions of each program to keep')
cmd:option('-useOneHot',false,'Represent programs using one-hot / otherwise use look-up-table')
cmd:option('-learningRate',1e-3,'learning rate')
cmd:option('-nEpochs',20,'training epochs')
cmd:option('-nConvLayers',1,'number of extra convolutional layers')
cmd:option('-nFCLayers',1,'number of extra convolutional layers')
cmd:option('-batchSize',1,'size of batch used in training')
cmd:option('-usemom',false,'use momentum during SGD optimisation')
cmd:option('-useRMSProp',false,'use alternative optimizer rather than SGD')
cmd:option('-useCUDA',true,'use CUDA optimisation')
cmd:option('-gpuid',1,'which GPU to use')
cmd:option('-usePreTrainedEmbedding',false,'initialise network with pre-trained embedding')
cmd:option('-fixEmbedding',false,'prevent the embedding from being updated during learning')

cmd:option('-programLen',8,'how many instructions to read')

cmd:option('-debug',false,'enter debug mode')

cmd:option('-dataAugProb',0.1,'probability of changing an instruction during data augmentation')
cmd:option('-dataAugMethod',1,'1 - substitue the semantically most similar instruction, 2 - substitue random instruction')

cmd:option('-trainingSetSize',2,'restrict the size of the training-set for evaluation purposes')
cmd:option('-markFunctionEnds',false,'place a marker at the end of each method which may help classification work better')

cmd:option('-saveModel',false,'save the model and data split')
cmd:option('-saveFileName','detect_malware_cnn','filename to save the network')

cmd:option('-decayLearningRate',false,'reduce learning rate by factor of 10 every so often')
cmd:option('-weightDecay',0,'weight decay for L2 regularisation')
cmd:option('-weightDecayFrac',0.1,'amount to reduce learning rate by, 0.1 or 0.5 are good values')

-- try using dropout in various places of the network
cmd:option('-useSpatialDropout',false,'drop instructions after the embedding layer')
cmd:option('-useDropoutAfterEmbedding',false,'drop instructions after the embedding layer')
cmd:option('-useDropoutAfterConv',false,'drop instructions after the embedding layer')

cmd:option('-dataDir','./malwareDataset/','directory with the android programs to classify')
cmd:option('-modelFile','./model.th7','file contained the pre-trained model')
cmd:option('-setupMode',false,'Only run in this mode once. Splits the data into the train/test sets. Saved into ./config/metaData.th7')

cmd:option('-maxSequenceLength',1000000,'if program is longer than this length, crop sequence before passing to GPU')

cmd:option('-dataAugTesting',false,'Use data augmentation during testing i.e average score over random samples from program')

opt = cmd:parse(arg)

if opt.useCUDA then
	require 'cunn'
	require 'cutorch'
end

torch.setdefaulttensortype("torch.DoubleTensor")
torch.manualSeed(opt.seed)
if opt.useCUDA then 
	cutorch.setDevice(opt.gpuid)
	cutorch.manualSeedAll(opt.seed)
end

if opt.dataAugTesting then
	require 'testModel_dataAug'
else
	require 'testModel'
end

print(opt)

function isnan(z)
	return z ~= z
end


--print(opt.metaDataFile)
local metaData = torch.load(opt.metaDataFile)

--print('reading data from disk')
local allData = readMalwareData(opt.dataDir,metaData)

testInds = {}
for i=1, labels:size(1) do
    testInds[i] = i
end

local model = torch.load(opt.modelFile)
local scores,valResult,valConfMat,valTime = testModel(allData,model,testInds)

for score in #scores:
    print(score)
