clc; 
tic
%Load training and testing datasets and balance classes
file = fullfile('jSRI_8n\\Blue\\train');
ds = imageDatastore(file,'IncludeSubfolders',true,'LabelSource','foldernames');
minSetCount = min(ds.countEachLabel{:,2}); 
trainSet = splitEachLabel(ds, minSetCount, 'randomize');
file = fullfile('jSRI_8n\\Blue\\test');
ds = imageDatastore(file,'IncludeSubfolders',true,'LabelSource','foldernames');
minSetCount = min(ds.countEachLabel{:,2}); 
testSet = splitEachLabel(ds, minSetCount, 'randomize');
%CNN
layers = [imageInputLayer([18 18 1]);
          convolution2dLayer(3,6,'Stride',1,'Padding',0); reluLayer();
          convolution2dLayer(3,18,'Stride',1,'Padding',0); reluLayer();
          maxPooling2dLayer(2,'Stride',2,'Padding',0);
          convolution2dLayer(3,36,'Stride',1,'Padding',0); reluLayer();
          convolution2dLayer(3,54,'Stride',1,'Padding',0); reluLayer();
          maxPooling2dLayer(2,'Stride',2,'Padding',0);
%           convolution2dLayer(3,72,'Stride',1,'Padding',0); reluLayer();
%           convolution2dLayer(3,90,'Stride',1,'Padding',0); reluLayer();
%           maxPooling2dLayer(2,'Stride',2);
          dropoutLayer();
          fullyConnectedLayer(5);
          softmaxLayer();
          classificationLayer()];
options = trainingOptions('sgdm','MaxEpochs',15,'InitialLearnRate',0.1, ...
                            'LearnRateSchedule', 'piecewise', ...
                            'LearnRateDropFactor', 0.2, ...
                            'LearnRateDropPeriod', 5, ...
                            'MiniBatchSize',500, ...
                            'OutputFcn',@plotTrainingAccuracy);  
convnet = trainNetwork(trainSet,layers,options);
%Testing
YTest = classify(convnet,testSet);
TTest = testSet.Labels;
accuracy = sum(YTest == TTest)/numel(TTest);
disp(['Accuracy = ' num2str(accuracy)]);
toc

im = imread('jSRI_8n\\Blue\\test\\1\\7_15290.png');
act1 = activations(convnet,im,'conv_1','OutputAs','channels');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
montage(mat2gray(act1),'Size',[2 3]);

%Accuracy Plot
function plotTrainingAccuracy(info)
persistent plotObj
if info.State == 'start'
    clf;
    plotObj = animatedline;
    xlabel('Iteration')
    ylabel('Training Accuracy')
elseif info.State == iteration'
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
end
load('plots.mat');
a(end+1)=info.TrainingAccuracy;
l(end+1)=double(gather(info.TrainingLoss));
save('plots.mat','a','l');
end