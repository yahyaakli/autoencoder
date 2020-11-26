%by Yahya AKLI
close all
% clear all
clc

U = load('MNISTnumImages5000_balanced.txt');

%indexes = round(5000*rand(100,1));

indexes = [];
for j=0:9
    indexes = [indexes ; [1:10]+500*j];
end

% U100 = U(indexes,:);
noise = 0.4*randn(size(U));
Un = zeros(size(U));
for i=1:size(U,1)
    for j=1:size(U,2)
        Un(i,j) = ((U(i,j)+noise(i,j))>0)*(min(U(i,j)+noise(i,j),1));
    end
end
U100 = Un(indexes,:);
for i=1:10
    for j = 1:10
        v = reshape(U100((i-1)*10+j,:),28,28);
        subplot(10,10,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
Ulabel = load('MNISTnumLabels5000_balanced.txt');


%% training & testing sets
% P = 0.80 ;
% m = length(Un)/10;
%
% Training = [];
% TrainingN = [];
% Training_labels = [];
% Testing =[];
% TestingN =[];
% Testing_labels = [];
% for i=0:9
%     idx = i*500 + randperm(m);
%     Training = [Training;U(idx(1:round(P*m)),:)] ;
%     TrainingN = [TrainingN;Un(idx(1:round(P*m)),:)] ;
%     Training_labels = [Training_labels;Ulabel(idx(1:round(P*m)),:)] ;
%     Testing = [Testing;U(idx(round(P*m)+1:length(idx)),:)];
%     TestingN = [TestingN;Un(idx(round(P*m)+1:length(idx)),:)];
%     Testing_labels = [Testing_labels;Ulabel(idx(round(P*m)+1:length(idx)),:)];
% end;

Training = [];
TrainingN = [];
Testing = [];
TestingN = [];

Training_labels = [];
Testing_labels = [];

for i=0:9
    Training = [Training;U(i*500+1:i*500+400,:)];
    TrainingN = [TrainingN;Un(i*500+1:i*500+400,:)];
    Training_labels = [Training_labels;Ulabel(i*500+1:i*500+400,:)];
    Testing = [Testing;U(i*500+401:(i+1)*500,:)];
    TestingN = [TestingN;Un(i*500+401:(i+1)*500,:)];
    Testing_labels = [Testing_labels;Ulabel(i*500+401:(i+1)*500,:)];
end

idx = randperm(length(Training));
Training = Training(idx,:);
TrainingN = TrainingN(idx,:);
Training_labels = Training_labels(idx,:);

idx = randperm(length(Testing));
Testing = Testing(idx,:);
TestingN = TestingN(idx,:);
Testing_labels = Testing_labels(idx,:);


%% auto encoder
nblayers = 1;
nbneurons = 200;

epochs = 200;
alpha = 0.3;
eta = 0.1;
[w1auto, w2auto, b1auto , b2auto,J , Jpn, ERR] = oneLayerAutoEncoder(TrainingN,Training,Training_labels,nbneurons,epochs,eta,alpha);

%% Performance tests
Jpntest = zeros(1,10);
Jtest = 0;
images = [];
for i=1:1000
    X = TestingN(i,:);
    y = Testing_labels(i,:);
    h2 = oneLayertest(X,w1auto, w2auto, b1auto , b2auto);
    images=[images;h2];
    Jq = 0.5*sum((X - h2).^2);
    Jtest = Jtest +Jq;
    Jpntest(y+1) = Jpntest(y+1)+Jq;
end

%% plot performance graphs
c = categorical({'training','testing'});
figure;bar(c,[J;Jtest],'c');
title('J2 loss');
figure;bar(0:1:9,[Jpn;Jpntest]')
legend('training','testing')
title('error for each digit');
%% plot error J2
Xx = 0:10:200;
figure;
plot(Xx,ERR,'b');
title('J2 loss for each 10th epoch');

%% plot random features
IM = randi(200,1,20);
figure;
for i=1:20
    v = reshape(w1auto(:,IM(i))',28,28);
    subplot(5,4,i);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
end
%% plot 8 random pictures vs prediction of auto encoder
IM = randi(1000,1,8);
figure;
for i=1:8
    X = Testing(IM(i),:);
    h2 = oneLayertest(X,w1auto, w2auto, b1auto , b2auto);
    
    v = reshape(X,28,28);
    subplot(2,8,i);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
    
    v = reshape(h2,28,28);
    subplot(2,8,8+i);
    image(64*v);
    colormap(gray(64));
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
    
end

%% problem 2
%% case 2
epochs = 200;
eta = 0.2 ;
nbclasses = 10;
[w1, w2, b1 , b2 ,ACC] = oneLayerNetwork(Training,Training_labels,nbclasses,nbneurons,epochs,eta,w1auto,b1auto);
%% testing
classes = [];
for i=1:1000
    classes=[classes,oneLayertest(Testing(i,:),w1,w2,b1,b2)'];
end

%% training accuracy
Xx = 0:10:200;
figure;
plot(Xx,ACC,'g',Xx,1-ACC,'b','LineWidth',2);
legend('accuracy','error');
title('balanced accuracy and error on training set');

%% confusion matrix
M = confusion_matrix(Training,Training_labels,w1,w2,b1,b2);
M_test = confusion_matrix(Testing,Testing_labels,w1,w2,b1,b2);

%% auto encoder from problem 1
epochs = 400;
alpha = 0.4;
eta = 0.2;
[w1auto, w2auto, b1auto , b2auto,J , Jpn, ERR] = oneLayerAutoEncoder(Training,Training,Training_labels,nbneurons,epochs,eta,alpha);

%% case 1
epochs = 200;
eta = 0.2 ;
nbclasses = 10;
[w1, w2, b1 , b2 ,ACC] = oneLayerNetwork(Training,Training_labels,nbclasses,nbneurons,epochs,eta,w1auto,b1auto);
%% testing
classes = [];
for i=1:1000
    classes=[classes,oneLayertest(Testing(i,:),w1,w2,b1,b2)'];
end

%% training accuracy
Xx = 0:10:200;
figure;
plot(Xx,ACC,'g',Xx,1-ACC,'b','LineWidth',2);
legend('accuracy','error');
title('balanced accuracy and error on training set');

%% confusion matrix
M = confusion_matrix(Training,Training_labels,w1,w2,b1,b2);
M_test = confusion_matrix(Testing,Testing_labels,w1,w2,b1,b2);