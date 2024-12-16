function [delta] = getGrad(H,S,alpha,Kneighbor)
%GRADGET 计算出给定S时，对比损失函数的梯度
%   此处显示详细说明
delta1=-2*H*(H')+2*H*(H')*S;
delta2=zeros(size(S));
[n,~]=size(H);
[~,k]=size(Kneighbor);
for i=1:n
    Kt=0;
    for p=1:n
        if p==i
            continue;
        end
        Kt=Kt+exp(S(i,p));
    end
    for j=1:n
        if ismember(j,Kneighbor(i,:))
            delta2(i,j)=-1+k*exp(S(i,j))/Kt;
        elseif i==j
            delta2(i,j)=0;
        else
            delta2(i,j)=k*exp(S(i,j))/Kt;
        end
    end
end
delta=delta1+alpha*delta2;
end

