function S = overlap(z,m)

e(5:5)=0;
e(1,1)=z(1);
for i=2:5
e(i,1)=e(i-1,1)+z(i);
end
for n=2:5
for k=2:n
e(n,k)=e(n-1,k)+z(n)*e(n-1,k-1);
end
end
S=e(5,m);