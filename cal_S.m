function [S] = cal_S( trainData,trainLabel,alphaCoeff,knnType,nuclass,i)
class = unique(trainLabel,'stable');
len = length(class);
S = zeros(length(trainLabel),length(trainLabel));
for ii = 1:len
    iclass = class(ii);
    index1 = find(trainLabel==iclass);
    trainData1 = trainData(index1,:);
    train1Num=length(index1);
    if knnType==0
        knn=ceil(sqrt(train1Num));%近邻数
    elseif knnType==-1
        knn=ceil(sqrt(train1Num)/2);
    elseif knnType==-2
        knn=ceil(sqrt(train1Num)*2);
    else
        knn=knnType;
    end
    Kneighbor = getKneighbors(trainData1,knn);
    [Si,f] = Adam(trainData1,alphaCoeff,0.001,Kneighbor);%H,alphaCoeff,k,alpha
    Si=mapminmax(Si,0,1);
    S(index1,index1)=Si;
end
end