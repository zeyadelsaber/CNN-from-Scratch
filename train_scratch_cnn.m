clear;
clc;
close all;

rng(1);
%% Load datasets
paths = project_paths();
trainPath = paths.training;
valPath = paths.validation;
testPath = paths.test;

imdsTrain = imageDatastore(trainPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsVal = imageDatastore(valPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

imdsTest = imageDatastore(testPath, ...
    "IncludeSubfolders",true, ...
    "LabelSource","foldernames");

%% Check classes and number of images
disp("Training set:");
countEachLabel(imdsTrain)

disp("Validation set:");
countEachLabel(imdsVal)

disp("Test set:");
countEachLabel(imdsTest)

img = readimage(imdsTrain,1);
disp("Image size:");
disp(size(img));

%% Show sample training images
figure;
perm = randperm(numel(imdsTrain.Files),9);

for i = 1:9
    subplot(3,3,i);
    sampleImg = readimage(imdsTrain,perm(i));
    imshow(sampleImg);
    title(string(imdsTrain.Labels(perm(i))));
end

sgtitle("Sample Training Images");
%%
augmenter = imageDataAugmenter( ...
    "RandRotation",[-10 10], ...
    "RandXTranslation",[-10 10], ...
    "RandYTranslation",[-10 10], ...
    "RandXReflection",true);

augTrain = augmentedImageDatastore( ...
    [300 300], ...
    imdsTrain, ...
    "DataAugmentation",augmenter);
augVal = augmentedImageDatastore([300 300],imdsVal);

augTest = augmentedImageDatastore([300 300],imdsTest);

%% Build simple CNN from scratch

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    imageInputLayer([300 300 3])

    convolution2dLayer(3, 16, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 32, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    convolution2dLayer(3, 64, "Padding", "same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
    ];

%analyzeNetwork(layers)

if canUseGPU
    execEnv = "gpu";
else
    execEnv = "cpu";
end

options = trainingOptions("adam", ...
    MaxEpochs = 10, ...
    MiniBatchSize = 32, ...
    InitialLearnRate = 1e-3, ...
    ValidationData = augVal, ...
    ValidationFrequency = 10, ...
    ValidationPatience = 5, ...
    Shuffle = "every-epoch", ...
    ExecutionEnvironment = execEnv, ...
    Plots = "training-progress", ...
    Verbose = true);
%% Train the CNN


netCNN = trainNetwork(augTrain, layers, options);

%% Test the CNN
YPredCNN  = classify(netCNN, augTest);

YTest = imdsTest.Labels;

accuracyCNN = mean(YPredCNN == YTest);
fprintf("Scratch CNN test accuracy: %.2f%%\n", accuracyCNN * 100);

figure
confusionchart(YTest, YPredCNN );
title("Scratch CNN Confusion Matrix");
exportgraphics(gcf, fullfile(paths.figures, "scratch_cnn_confusion_matrix.png"));

%%save

save(fullfile(paths.models, "scratch_cnn_result.mat"), ...
    "netCNN","accuracyCNN","YPredCNN","YTest");

