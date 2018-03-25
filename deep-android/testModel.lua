function testModel(allData,model)

	--print('testing corrected verison 2')

	local timerTest = torch.Timer()

	local dtype = 'torch.DoubleTensor'
	if opt.useCUDA then
		dtype = 'torch.CudaTensor'
	end

	local criterion = nn.ClassNLLCriterion():type(dtype)

	model:evaluate()
	
	-- push the validation data through the network
	local nValPrograms = #allData.label
	local valError = 0
	local correct = 0
	local confmat = torch.zeros(2,2)
	local scores = torch.zeros(nValPrograms, 2)

	-- We need to make sure the rare-class is regarded as positive
	-- This means the f-score etc will be corectly calculated
	-- When reading the data benign is labelled as 1 and malware as 2
	local nBenign = 0
	local nMalware = 0
	for k = 1,nValPrograms do
		if allData.label[k] == 1 then
			nBenign = nBenign + 1
		else
			nMalware = nMalware + 1
		end
	end
	local positiveLabel = 1
	if nMalware < nBenign then
		positiveLabel = 2
	end

	print('Test Stats : nMalware ',nMalware, ' nBenign ',nBenign, ' positiveLabel ',positiveLabel)

	--local valBatch = torch.zeros(1,opt.programLen):type(dtype)
	local valLabel = torch.zeros(1):type(dtype)

	for k = 1,nValPrograms do
		valLabel[{1}] = allData.label[k]
		--valBatch[{{1},{}}] = allData.program[k]

		local currProgramPtr = allData.programStartPtrs[k]
		local currProgramLen = allData.programLengths[k]

		if currProgramLen > opt.maxSequenceLength then
			currProgramLen = opt.maxSequenceLength
		end			

		local valBatch = torch.zeros(1,currProgramLen):type(dtype)
		valBatch[{{1},{}}] = allData.program[{{currProgramPtr,currProgramPtr + currProgramLen - 1}}]

		local netOutput = model:forward(valBatch)

		valError = valError + criterion:forward(netOutput,valLabel)	 		
		local netOutputProb = nn.Exp():forward(netOutput:double())
		scores[k] = netOutputProb

		local v,i = torch.max(netOutputProb,2)
		local pred = i[{1,1}]
		local gt = allData.label[k]
		if pred == gt then
			correct = correct + 1;
		end
		confmat[pred][gt] = confmat[pred][gt] + 1
	end
	valError = valError / nValPrograms

	local tp = 0
	local fp = 0
	local fn = 0

	if positiveLabel == 1 then
		tp = confmat[1][1]
		fp = confmat[1][2]
		fn = confmat[2][1]
	else
		tp = confmat[2][2]
		fp = confmat[2][1]
		fn = confmat[1][2]
	end

	local testResult = {
		-- tp = tp,
		-- fp = fp,
		-- fn = fn,
		prec = tp / (tp + fp),
		recall = tp / (tp + fn),
		fscore = (2 * tp) / ((2 * tp) + fp + fn),
		accuracy = correct/nValPrograms,
		testError = valError,		
	}

	local time = timerTest:time().real	

	model:training()

	-- clean up
	valLabel = nil
	collectgarbage()

	return scores,testResult,confmat,time
end