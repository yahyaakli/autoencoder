function M = confusion_matrix(Training,label,w1,w2,b1,b2)
M = zeros(10,10);
for k=1:size(Training,1)
    class = oneLayertest(Training(k,:),w1,w2,b1,b2);
    predicted_label = find(class==1)-1;
    M(label(k)+1,predicted_label+1)=M(label(k)+1,predicted_label+1)+1;
end
end