function [Kneighbor] = getKneighbors(x,k)
%GETKNEIGHBOR x为训练集中同一类（正或负）样本的特征:n1_p*(d-1),
%k为近邻数，Kneighbor具体的k近邻的样本:n*k
%x=[1,1;1,3;4,3;1,2;3,2;3,1;3,1];k=2;
[n,~] = size(x); % 样本数量
% 使用pdist2计算所有样本对之间的距离矩阵
D = pdist2(x,x)-eye(n);
% 将距离矩阵转换为邻接矩阵，然后找到每个点的前k个最小距离（不包括自身）
[~, Kneighbor] = sort(D, 'ascend'); % 对距离矩阵进行升序排序
Kneighbor = Kneighbor(2:(k+1),:)'; % 去掉第一列（即每个样本到自身的距离）
end
