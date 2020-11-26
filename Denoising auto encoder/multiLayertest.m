function [h2] = multiLayertest(X,w1, wout, whidden, b1, bout, bhidden)
f=@(u)(1./(1+exp(-u)));
nblayers = size(whidden,1);
nbneurons = size(wout,1);
% feed-forward
% input layer
s1 = X*w1+b1;
h1 = f(s1);

%hidden layers
H = [h1];
S = [s1];
for i=1:nblayers
    W = reshape(whidden(i,:),nbneurons,nbneurons);
    sj = H(end,:)*W+bhidden(i,:);
    S = [S;sj];
    H = [H;f(sj)];
end

%output layer
s2 = H(end,:)*wout+bout;
h2 = threshold(f(s2));
end