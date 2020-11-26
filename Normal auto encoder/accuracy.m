function acc = accuracy(Training,label,w1,w2,b1,b2)
f=@(u)(1./(1+exp(-u)));
acc = 0;
for k=1:size(Training,1)
    X = Training(k,:);
    y = label(k);
    s1 = X*w1+b1;
    h1 = f(s1);
    s2 = h1*w2+b2;
    h2 = threshold(f(s2));
    h2(h2<0.75) = 0;
    yout = output_vector(y);
    if(sum(abs(yout-h2)) == 0)
        acc=acc+1;
    end
end
acc=acc/size(Training,1);
end