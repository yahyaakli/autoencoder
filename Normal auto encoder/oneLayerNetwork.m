function [w1, w2, b1 , b2 ,ACC] = oneLayerNetwork(Training,label,nbclasses,nbneurons,epochs,eta)
f=@(u)(1./(1+exp(-u)));
df=@(u)((1-f(u)).*f(u));
N = size(Training,2);

w1 = 0.01*rand(N,nbneurons);
b1 = zeros(1,nbneurons);
w2 = 0.01*rand(nbneurons,nbclasses);
b2 = zeros(1,nbclasses);
ACC = [];
for e=1:epochs
    J = 0;
    for k=1:size(Training,1)
        X = Training(k,:);
        y = label(k); %label
        % feed-forward
        s1 = X*w1+b1;
        h1 = f(s1);
        s2 = h1*w2+b2;
        h2 = f(s2); % network output
        
        % backpropagation
        % output layer
        yout = output_vector(y);
        dJout = df(s2).*(yout - h2);
        Jq = 0.5*sum((yout - h2).^2);
        J = J+Jq;
        %hidden layer
        dhidden = (df(s1)').*(w2*dJout');
        
        %update weights
        w2 = w2 + eta*h1'*dJout;
        w1 = w1 + eta*X'*dhidden';
        
        %update biases
        b2 = b2 + eta*dJout;
        b1 = b1 + eta*dhidden';
    end
    acc = accuracy(Training,label,w1,w2,b1,b2);
    if(mod(e,10)==0 || e == 1)
       ACC = [ACC,acc]; 
    end
    fprintf("number of epochs reached %i, the accuracy in this epoch is %i\n", e, acc);
    if(J<0.03)
        break;
    end
    
end

end