% Array z (100x10) contains all random amplitudes, 1-5 for spin up and 6-10
% for spin down. Actually they are the same, e.g. z(1,3)=z(1,8), 
% but this script can be used even if these amplitudes are differrent 

for n=1:N
for m=1:N
z1=z(:,n);
z2=z(:,m);
    
    
z1u=z1(1:5);
z1d=z1(6:10);
z2u=z2(1:5);
z2d=z2(6:10);

% Function "overlap" is used separately for spin-up and spin-down. 
Su=overlap(z1u.*z2u,3);
Sd=overlap(z1d.*z2d,3);
S(n,m)=Su*Sd;

F1(1:10,1:10)=0;
F2(1:10,1:10,1:10,1:10)=0;

%One-electron matrix elements
for i=1:5
for j=1:5
z1u=z1(1:5);
z2u=z2(1:5);
F1(i,j)=z1u(i)*z2u(j);
z1u(i)=0;
z2u(j)=0;
z1u(1:i)=-z1u(1:i);
z2u(1:j)=-z2u(1:j);
F1(i,j)=F1(i,j)*overlap(z1u.*z2u,2)*Sd;
end
end

for i=1:5
for j=1:5
z1d=z1(6:10);
z2d=z2(6:10);
F1(i+5,j+5)=z1d(i)*z2d(j);
z1d(i)=0;
z2d(j)=0;
z1d(1:i)=-z1d(1:i);
z2d(1:j)=-z2d(1:j);
F1(i+5,j+5)=F1(i+5,j+5)*overlap(z1d.*z2d,2)*Su;
end
end

% Array I1 contains one-electron integrals 
E1(n,m)=sum(sum(F1.*I1));

%Two-electron matrix elements
for i1=1:5
for i2=i1+1:5
for j1=1:5
for j2=j1+1:5
z1u=z1(1:5);
z2u=z2(1:5);
F2(i1,i2,j2,j1)=z1u(i1)*z1u(i2)*z2u(j1)*z2u(j2);
z1u(i1)=0;
z2u(j1)=0;
z1u(i2)=0;
z2u(j2)=0;
z1u(i1:i2)=-z1u(i1:i2);
z2u(j1:j2)=-z2u(j1:j2);
F2(i1,i2,j2,j1)=F2(i1,i2,j2,j1)*overlap(z1u.*z2u,1)*Sd;
F2(i2,i1,j2,j1)=-F2(i1,i2,j2,j1);
F2(i1,i2,j1,j2)=-F2(i1,i2,j2,j1);
F2(i2,i1,j1,j2)=F2(i1,i2,j2,j1);
end
end
end
end

for i1=1:5
for i2=i1+1:5
for j1=1:5
for j2=j1+1:5
z1d=z1(6:10);
z2d=z2(6:10);
F2(i1+5,i2+5,j2+5,j1+5)=z1d(i1)*z1d(i2)*z2d(j1)*z2d(j2);
z1d(i1)=0;
z2d(j1)=0;
z1d(i2)=0;
z2d(j2)=0;
z1d(i1:i2)=-z1d(i1:i2);
z2d(j1:j2)=-z2d(j1:j2);
F2(i1+5,i2+5,j2+5,j1+5)=F2(i1+5,i2+5,j2+5,j1+5)*overlap(z1d.*z2d,1)*Su;
F2(i2+5,i1+5,j2+5,j1+5)=-F2(i1+5,i2+5,j2+5,j1+5);
F2(i1+5,i2+5,j1+5,j2+5)=-F2(i1+5,i2+5,j2+5,j1+5);
F2(i2+5,i1+5,j1+5,j2+5)=F2(i1+5,i2+5,j2+5,j1+5);
end
end
end
end

for i1=1:5
for i2=6:10
for j1=1:5
for j2=6:10
z1u=z1(1:5);
z1d=z1(6:10);
z2u=z2(1:5);
z2d=z2(6:10);
F2(i1,i2,j2,j1)=z1u(i1)*z1d(i2-5)*z2u(j1)*z2d(j2-5);
z1u(i1)=0;
z2u(j1)=0;
z1d(i2-5)=0;
z2d(j2-5)=0;
z1u(1:i1)=-z1u(1:i1);
z2u(1:j1)=-z2u(1:j1);
z1d(1:i2-5)=-z1d(1:i2-5);
z2d(1:j2-5)=-z2d(1:j2-5);
F2(i1,i2,j2,j1)=F2(i1,i2,j2,j1)*overlap(z1u.*z2u,2)*overlap(z1d.*z2d,2);
F2(i2,i1,j2,j1)=-F2(i1,i2,j2,j1);
F2(i1,i2,j1,j2)=-F2(i1,i2,j2,j1);
F2(i2,i1,j1,j2)=F2(i1,i2,j2,j1);
end
end
end
end

% Array I2 (10x10x10x10) contains two-electron integrals 
E2(n,m)=sum(sum(sum(sum(F2.*I2))));

end
end

% Hamiltonian. Constant 1.8 here is nuclear repulsion. 
H=E1+E2/2+1.8*S;