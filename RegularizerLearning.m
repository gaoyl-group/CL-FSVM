function [G,testFweight1] = RegularizerLearning(train_data,train_label,svm,kertype,S,lambda,m,knnType,Fweight1,paraLambda)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
label=unique(train_label,'stable');
labelNum=length(label);
[n,~]=size(train_label);
G=zeros(n,1);
testFweight1 = zeros(n,1);

for i=1:labelNum
    class1=label(i,:);
    a=find(train_label==class1);
    train1=train_data(a,:);
    [train1Num,~]=size(train1);
    if knnType==0
        knn=ceil(sqrt(train1Num));%近邻数
    elseif knnType==-1
        knn=ceil(sqrt(train1Num)/2);
    elseif knnType==-2
        knn=ceil(sqrt(train1Num)*2);
    else
        knn=knnType;
    end
    Kneighbor = getKneighbors(train1,knn);
    Si=S(a,a);
    for j=1:train1Num
        kneigh=Kneighbor(j,:);
        testFweight1(a(j),1)=lambda*Fweight1(a(j),1)+(1-lambda)*Si(j,kneigh)*Fweight1(kneigh);
        for k=kneigh
            y=svmTest_multiclass(svm, train_data(a(k),:)', kertype, paraLambda).score;

            if y>0
                G(a(j),1)=G(a(j),1)+Si(j,k)*(y^m);
            elseif y<0
                G(a(j),1)=G(a(j),1)-Si(j,k)*((-y)^m);
            end

            %G(a(j),1)=G(a(j),1)+Si(j,k)*y*lambda;

            %G(a(j),1)=G(a(j),1)+m*Si(j,k)*exp(y*0.01);
        end

    end
end

