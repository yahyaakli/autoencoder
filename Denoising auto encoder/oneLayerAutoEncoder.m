function [w1, w2, b1 , b2 ,J , Jpn, ERR] = oneLayerAutoEncoder(Training,U,label,nbneurons,epochs,eta,alpha)
f=@(u)(1./(1+exp(-u)));
df=@(u)((1-f(u)).*f(u));
N = size(Training,2);
nbclasses = N;
w1 = 0.01*rand(N,nbneurons);
b1 = zeros(1,nbneurons);
w2 = 0.01*rand(nbneurons,nbclasses);
b2 = zeros(1,nbclasses);
ERR=[];
for e=1:epochs
    J = 0;
    Jpn = zeros(1,10);
    Dws = {zeros(N,nbneurons),zeros(nbneurons,N)};
    for k=1:size(Training,1)
        X = Training(k,:);
        Ux = U(k,:);
        y = label(k); %label
        % feed-forward
        s1 = X*w1+b1;
        h1 = f(s1);
        s2 = h1*w2+b2;
        h2 = f(s2); % network output
        
        % backpropagation
        % output layer
        
        dJout = df(s2).*(Ux - h2);
        
        Jq = 0.5*sum((Ux - h2).^2);
        J = J+Jq;
        Jpn(y+1) = Jpn(y+1)+Jq;
        %hidden layer
        dhidden = (df(s1)').*(w2*dJout');
        
        %update weights
        w2 = w2 + (eta*h1'*dJout+alpha*Dws{2});
        w1 = w1 + (eta*X'*dhidden'+alpha*Dws{1});
        
        %update biases
        b2 = b2 + eta*dJout;
        b1 = b1 + eta*dhidden';
        
        Dws{2} = eta*h1'*dJout+alpha*Dws{2};
        Dws{1} = eta*X'*dhidden'+alpha*Dws{1};
    end
    if(mod(e,10)==0 || e == 1)
        ERR = [ERR,J];
    end
    fprintf("number of epochs reached %i, the loss function value %i\n", e, J);
    if(J<2e+03)
        break;
    end
    
end

end