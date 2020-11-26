function [h2,Jq] = oneLayertest(X,w1,w2,b1,b2)
f=@(u)(1./(1+exp(-u)));
s1 = X*w1+b1;
h1 = f(s1);
s2 = h1*w2+b2;
h2 = threshold(f(s2));
h2(h2<0.75) = 0;
end