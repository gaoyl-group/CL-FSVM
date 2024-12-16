function [S] = initialTheta(H,alpha)
%INITIALTHETA 此处显示有关此函数的摘要
%   此处显示详细说明
[len,~]=size(H);
S=(H*H'+alpha*eye(len))\(H*H');
S=(S+S')/2;
end

