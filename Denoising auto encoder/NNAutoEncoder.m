function [w1, wout, whidden, b1, bout, bhidden,J , Jpn, ERR] = NNAutoEncoder(Training,U,label,nbclasses,nblayers,nbneurons,epochs,eta,alpha)
    if (nblayers == 1)
        [w1, wout, b1 , bout ,J , Jpn, ERR] = oneLayerNetwork(Training,U,label,nbclasses,nbneurons,epochs,eta,alpha);
        whidden = [];
        bhidden = [];
    else
        [w1, wout, whidden, b1, bout, bhidden,J , Jpn, ERR] = multiLayerNetwork(Training,U,label,nbclasses,nblayers,nbneurons,epochs,eta,alpha);
    end
end