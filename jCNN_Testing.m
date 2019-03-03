tic
load('convnet_blue.mat');
I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\A\\%s_%d_%d-%d.mat','Black',1,1,2));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum((GT-5)==0)))/sum(sum(GT>0)); fprintf('A: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\B\\%s_%d_%d-%d.mat','Black',1,1,2));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('B: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\C\\%s_%d_%d-%d.mat','Black',1,1,2));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('C: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\D\\%s_%d_%d-%d.mat','Black',1,1,2));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('D: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\E\\%s_%d_%d-%d.mat','Black',1,1,2));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('E: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\F\\%s_%d_%d-%d-%d.mat','Black',1,1,2,3));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('F: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\G\\%s_%d_%d-%d-%d-%d.mat','Black',1,1,2,3,4));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('G: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');

I=uint8(zeros(81,627));
load(sprintf('jCombosCNN\\H\\%s_%d_%d-%d-%d-%d-%d.mat','Black',1,1,2,3,4,5));
[SRV,RC]=hsi2srv(HSI);
for i=1:size(SRV,1)
    SRVimage=reshape([SRV(i,:) zeros(1,3)],6,6);
    I(RC(i,1),RC(i,2))=uint8(classify(convnet,SRVimage));
end
accuracy=100*(sum(sum(I==(GT-5)))-sum(sum(GT==0)))/sum(sum(GT>0)); fprintf('H: %f\n',accuracy);
figure; subplot(2,1,1); imshow(jColor(I)); xlabel('CNN Clustering Result');
subplot(2,1,2); imshow(jColor(GT)); xlabel('Ground Truth');
toc