function [w1, wout, whidden, b1, bout, bhidden] = multiLayerNetwork(Training,label,nbclasses,nblayers,nbneurons,epochs,eta)
f=@(u)(1./(1+exp(-u)));
df=@(u)((1-f(u)).*f(u));

N = size(Training,2);
iter = size(Training,1);
%initialize the weights and biases for the input layer
w1 = 0.1*rand(N,nbneurons);
b1 = zeros(1,nbneurons);

%initialize the weights and the bias for the hidden layers
whidden = 0.1*rand(nblayers,nbneurons*nbneurons);
bhidden = zeros(nblayers,nbneurons);

%initialize the weights and the bias for the output layers
wout = 0.1*rand(nbneurons,nbclasses);
bout = zeros(1,nbclasses);

for e=1:epochs
    for k=1:iter
        X = Training(k,:);
        y = label(k); %label
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
        
        %backpropagation
        % output layer
        yout = output_vector(y);
        dJout = df(s2).*(yout - h2);
        
        % hidden layers
        dtmp = (df(S(end,:))').*wout*dJout';
        
        DTMP = [dtmp];
        
        for a=nblayers:-1:1
            W = reshape(whidden(a,:),nbneurons,nbneurons);
            dtmp = (df(S(a,:))').*(W*dtmp);
            DTMP = [dtmp,DTMP];
        end
        
        % update weights and biases
        % output layer
        wout = wout + eta*H(end,:)'*dJout;
        bout = bout + eta*dJout;
        
        % hidden layer
        for a=nblayers:-1:1
            W = reshape(whidden(a,:),nbneurons,nbneurons);
            W = W+eta*H(a,:)'*DTMP(:,a)';
            whidden(a,:) = reshape(W,1,nbneurons*nbneurons);
            bhidden(a,:) = bhidden(a,:) + eta*DTMP(:,a)';
        end
        
        w1 = w1 + eta*X'*dtmp';
        b1 = b1 + eta*dtmp';
    end
    fprintf("number of epochs reached %i\n", e);
end
end