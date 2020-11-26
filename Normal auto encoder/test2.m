%Reads the text version of the 1000-number MNIST data file and displays the
%first 100 in a 10X10 array.
%
%Written by: Ali Minai(3/14/16)


U = load('MNISTnumImages5000_balanced.txt');

%indexes = round(5000*rand(100,1));

indexes = [];
for j=0:9
    indexes = [indexes ; [1:10]+500*j];
end

U100 = U(indexes,:);

% for i=1:10
%     for j = 1:10
%         v = reshape(U100((i-1)*10+j,:),28,28);
%         subplot(10,10,(i-1)*10+j)
%         image(64*v)
%         colormap(gray(64));
%         set(gca,'xtick',[])
%         set(gca,'xticklabel',[])
%         set(gca,'ytick',[])
%         set(gca,'yticklabel',[])
%         set(gca,'dataaspectratio',[1 1 1]);
%     end
% end
Ulabel = load('MNISTnumLabels5000_balanced.txt');


%% training & testing sets
P = 0.80 ;
m = length(U)/10;

Training = [];
Training_labels = [];
Testing =[];
Testing_labels = [];
for i=0:9
    idx = i*500 + randperm(m);
    Training = [Training;U(idx(1:round(P*m)),:)] ;
    Training_labels = [Training_labels;Ulabel(idx(1:round(P*m)),:)] ;
    Testing = [Testing;U(idx(round(P*m)+1:length(idx)),:)];
    Testing_labels = [Testing_labels;Ulabel(idx(round(P*m)+1:length(idx)),:)];
end;

%% shuffel the training & testing sets
m = length(Training);
idx = randperm(m);
Training = Training(idx,:);
Training_labels = Training_labels(idx,:);

m = length(Testing);
idx = randperm(m);
Testing = Testing(idx,:);
Testing_labels = Testing_labels(idx,:);


%% initializing parameters
nblayers = 1;
nbneurons = 200;

epochs = 200;
eta = 0.2 ;
nbclasses = 10;

nbinputs = 784;
smallTrain = Training(1:100,:);
smallLabel = Training_labels(1:100,:);

xtest = U(600,:);
ltest = Testing_labels(1,:);
Utrain = U(1:200,:);
Utrainlabel = Ulabel(1:200,:);
%%

[w1, w2, whidden, b1, b2, bhidden] = NNetwork(Training,Training_labels,nbclasses,nblayers,nbneurons,epochs,eta);

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

%% auto encoder
epochs = 400;
alpha = 0.4;
eta = 0.2;
[w1auto, w2auto, b1auto , b2auto,J , Jpn, ERR] = oneLayerAutoEncoder(Training,Training_labels,nbneurons,epochs,eta,alpha);

%% Performance tests
Jpntest = zeros(1,10);
Jtest = 0;
images = [];
for i=1:1000
    X = Testing(i,:);
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
Xx = 0:10:400;
figure;
plot(Xx,ERR,'b');
title('J2 loss for each 10th epoch');

%% plot randomly images
IM = randi(200,1,20);
figure;
for i=1:20
    v = reshape(w1(:,IM(i))',28,28);
    subplot(5,4,i);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
end

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
    v = reshape(images(IM(i),:),28,28);
    subplot(2,8,i);
    image(64*v);
    colormap(gray(64));
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
    v = reshape(Testing(IM(i),:),28,28);
    subplot(2,8,8+i);
    image(64*v);
    colormap(gray(64));
    set(gca,'xtick',[])
    set(gca,'xticklabel',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',[])
    set(gca,'dataaspectratio',[1 1 1]);
end
% for i=1:10
%     for j = 1:10
%         v = reshape(U100((i-1)*10+j,:),28,28);
%         subplot(10,10,(i-1)*10+j)
%         image(64*v)
%         colormap(gray(64));
%         set(gca,'xtick',[])
%         set(gca,'xticklabel',[])
%         set(gca,'ytick',[])
%         set(gca,'yticklabel',[])
%         set(gca,'dataaspectratio',[1 1 1]);
%     end
% end
