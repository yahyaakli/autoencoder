function [w1, wout, whidden, b1, bout, bhidden] = NNetwork(Training,label,nbclasses,nblayers,nbneurons,epochs,eta)
    if (nblayers == 1)
        [w1, wout, b1 , bout] = oneLayerNetwork(Training,label,nbclasses,nbneurons,epochs,eta);
        whidden = [];
        bhidden = [];
    else
        [w1, wout, whidden, b1, bout, bhidden] = multiLayerNetwork(Training,label,nbclasses,nblayers,nbneurons,epochs,eta);
    end
end