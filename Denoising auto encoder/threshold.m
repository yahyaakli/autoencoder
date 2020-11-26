 function y = threshold(a)
y = zeros(1,length(a));
for i=1:length(a)
   if(a(i)>0.75)
      y(i)=1;
   elseif (a(i)<0.25)
      y(i)=0;
   else
      y(i)=a(i);
   end
end
end