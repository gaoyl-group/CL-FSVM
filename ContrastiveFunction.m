function [f] = ContrastiveFunction(H,S,alpha,Kneighbor)
%CONTRASTIVEFUNCTION 计算出给定S时，对比损失函数的值
%H:n*d,S:n*n,k近邻数,Kneighbor是具体的k近邻的样本:n*k
n=size(S);
f1=norm(H'-H'*S,'fro')^2;
f2=0;
for i=1:n
    Kt=0;
    for p=1:n
        if p==i
            continue;
        end
        Kt=Kt+exp(S(i,p));
    end
    for j=Kneighbor(i,:)
        f2=f2-log(exp(S(i,j))/Kt);
    end
end
f=f1+alpha*f2;