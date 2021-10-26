function [Cluster_elem,M,EW_history,W,z]=CGFFCM(X,M,k,p_init,p_max,p_step,t_max,beta_memory,N,fuzzy_degree,d,beta_z,landa,v,G)
%
% This demo implements the CGFFCM algorithm as described in
% A.Golzari oskouei, M.Hashemzadeh, B.Asheghi  and M.Balafar, "CGFFCM: Cluster-weight 
% and Group-local Feature-weight learning in Fuzzy C-Means clustering algorithm for color
% image segmentation", Applied Soft Computing, 2021 (submited).
%
%Function Inputs
%===============
%
%X is an Nxd data matrix, where each row corresponds to an instance.
%
%M is a kxd matrix of the initial cluster centers. Each row corresponds to a center.
%
%k is the number of clusters.
%
%p_init is the initial p value (0<=p_init<1).
%
%p_max is the maximum admissible p value (0<=p_max<1).
%
%p_step is the step for increasing p (p_step>=0).
%
%t_max is the maximum number of iterations.
%
%Function Outputs
%================
%
%Cluster_elem is an N-dimensional row vector containing the final cluster assignments.
%Clusters are indexed 1,2,...,k.
%
%M is a kxd matrix of the final cluster centers. Each row corresponds to a center.
%
%Courtesy of A. Golzari

if p_init<0 || p_init>=1
    error('p_init must take a value in [0,1)');
end

if p_max<0 || p_max>=1
    error('p_max must take a value in [0,1)');
end

if p_max<p_init
    error('p_max must be greater or equal to p_init');
end

if p_step<0
    error('p_step must be a non-negative number');
end

if beta_memory<0 || beta_memory>1
    error('beta must take a value in [0,1]');
end

if beta_z==0
    error('beta must be a non-zero number');
end

if p_init==p_max
    
    if p_step~=0
        fprintf('p_step reset to zero, since p_max equals p_init\n\n');
    end
    
    p_flag=0;
    p_step=0;
    
elseif p_step==0
    
    if p_init~=p_max
        fprintf('p_max reset to equal p_init, since p_step=0\n\n');
    end
    
    p_flag=0;
    p_max=p_init;
    
else
    p_flag=1; %p_flag indicates whether p will be increased during the iterations.
end

%--------------------------------------------------------------------------

%Weights are uniformly initialized.
W=ones(1,k)/k;
z(:,1:3)=ones(k,3)/3;
z(:,4:6)=ones(k,3)/3;
z(:,7:8)=ones(k,2)/2;


%Other initializations.
p=p_init; %Initial p value.
p_prev=p-10^(-8); %Dummy value.
empty=0; %Count the number of iterations for which an empty or singleton cluster is detected.
Iter=1; %Number of iterations.
E_w_old=inf; %Previous iteration objective (used to check convergence).
Cluster_elem_history=[];
W_history=[];
z_history=[];

%--------------------------------------------------------------------------

fprintf('\nStart of CGFFCM iterations\n');
fprintf('----------------------------------\n\n');

%The CGFFCM iterative procedure.
nn=1;
while 1

        %Update the cluster assignments.
    for j=1:k
        distance(j,:,:) = (1-exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2))).*repmat(v,N,1);
        WBETA = transpose(z(j,:).^beta_z);
        WBETA(WBETA==inf)=0;
        dNK(:,j) = W(1,j).^p * reshape(distance(j,:,:),[N,d]) * WBETA   ;
    end
    
    tmp1 = zeros(N,k);
    for j=1:k
        tmp2 = (dNK./repmat(dNK(:,j),1,k)).^(1/(fuzzy_degree-1));
        tmp2(tmp2==inf)=0;
        tmp2(isnan(tmp2))=0;
        tmp1=tmp1+tmp2;
    end
    Cluster_elem = transpose(1./tmp1);
    Cluster_elem(isnan(Cluster_elem))=1;
    Cluster_elem(Cluster_elem==inf)=1;
    
    if nnz(dNK==0)>0
        for j=1:N
            if nnz(dNK(j,:)==0)>0
                Cluster_elem(find(dNK(j,:)==0),j) = 1/nnz(dNK(j,:)==0);
                Cluster_elem(find(dNK(j,:)~=0),j) = 0;
            end
        end
    end
    %Calculate the CGFFCM objective.
    E_w=object_fun(N,d,k,Cluster_elem,landa,M,fuzzy_degree,W,z,beta_z,p,X,v);
    EW_history(nn)= E_w;
    nn=nn+1;
    
    %If empty or singleton clusters are detected after the update.
    for i=1:k
        
        I=find(Cluster_elem(i,:)<=0.05);
        if length(I)==N-1 || length(I)==N
            
            fprintf('Empty or singleton clusters detected for p=%g.\n',p);
            fprintf('Reverting to previous p value.\n\n');
            
            E_w=NaN; %Current objective undefined.
            empty=empty+1;
            
            %Reduce p when empty or singleton clusters are detected.
            if empty>1
                p=p-p_step;
                
                %The last p increase may not correspond to a complete p_step,
                %if the difference p_max-p_init is not an exact multiple of p_step.
            else
                p=p_prev;
            end
            
            p_flag=0; %Never increase p again.
            
            %p is not allowed to drop out of the given range.
            if p<p_init || p_step==0
                
                fprintf('\n+++++++++++++++++++++++++++++++++++++++++\n\n');
                fprintf('p cannot be decreased further.\n');
                fprintf('Either p_step=0 or p_init already reached.\n');
                fprintf('Aborting Execution\n');
                fprintf('\n+++++++++++++++++++++++++++++++++++++++++\n\n');
                
                %Return NaN to indicate that no solution is produced.
                M=NaN(k,size(X,2));
                return;
            end
            
            %Continue from the assignments and the weights corresponding
            %to the decreased p value.
            a=(k*empty)-(k-1);
            b=k*empty;
            Cluster_elem=Cluster_elem_history(a:b,:);
            W=W_history(empty,:);
            aa=(k*empty)-(k-1);
            bb=k*empty;
            z=z_history(aa:bb,:);
            break;
        end
    end
    
    if ~isnan(E_w)
        fprintf('p=%g\n',p);
        fprintf('The CGFFCM objective is E_w=%f\n\n',E_w);
    end
    
    %Check for convergence. Never converge if in the current (or previous)
    %iteration empty or singleton clusters were detected.
    if ~isnan(E_w) && ~isnan(E_w_old) && (abs(1-E_w/E_w_old)<1e-6 || Iter>=t_max)
        
        fprintf('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n');
        fprintf('Converging for p=%g after %d iterations.\n',p,Iter);
        fprintf('The final CGFFCM objective is E_w=%f.\n',E_w);
        fprintf('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n');
        
        break;
        
    end
    
    E_w_old=E_w;
    
    %Update the cluster centers.
    mf = Cluster_elem.^fuzzy_degree;       % MF matrix after exponential modification
    for j=1:k
        M(j,:) = (mf(j,:) * (X .* (exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)))))./(((mf(j,:)*(exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)))))); %new center
    end
    
    %Increase the p value.
    if p_flag==1
        
        %Keep the assignments-weights corresponding to the current p.
        %These are needed when empty or singleton clusters are found in
        %subsequent iterations.
        Cluster_elem_history=[Cluster_elem;Cluster_elem_history];
        W_history=[W;W_history];
        z_history=[z;z_history];
        
        p_prev=p;
        p=p+p_step;
        
        if p>=p_max
            p=p_max;
            p_flag=0;
            fprintf('p_max reached\n\n');
        end
    end
    
    W_old=W;
    z_old=z;
    
    %Update the feature weights.
    for j=1:k
        distance(j,:,:) = (1-exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2)));
        dWkm(j,:) = (Cluster_elem(j,:).^fuzzy_degree) * reshape(distance(j,:,:),[N,d]);
    end
    
    tmp1 = zeros(k,d);
    for j=1:d
        tmp2 = (dWkm./repmat(dWkm(:,j),1,d)).^(1/(beta_z-1));
        tmp2(tmp2==inf)=0;
        tmp2(isnan(tmp2))=0;
        tmp1=tmp1+tmp2;
        if j==3
            z(:,1:3) = 1./tmp1(:,1:3);
            tmp1 = zeros(k,d);
            tmp2 = zeros(k,d);
        end
        if j==6
            z(:,4:6) = 1./tmp1(:,4:6);
            tmp1 = zeros(k,d);
            tmp2 = zeros(k,d);
        end
        if j==8
            z(:,7:8) = 1./tmp1(:,7:8);
            tmp1 = zeros(k,d);
            tmp2 = zeros(k,d);
        end
    end
    z(isnan(z))=1;
    z(z==inf)=1;
    
    if nnz(dWkm==0)>0
        for j=1:k
            if nnz(dWkm(j,:)==0)>0
                
                if find(dWkm(j,1:3)==0)
                    z(j,find(dWkm(j,1:3)==0)) = 1/nnz(dWkm(j,1:3)==0);
                    z(j,find(dWkm(j,1:3)~=0)) = 0;
                end
                
                if find(dWkm(j,4:6)==0)
                    z(j,find(dWkm(j,4:6)==0)) = 1/nnz(dWkm(j,4:6)==0);
                    z(j,find(dWkm(j,4:6)~=0)) = 0;
                end
                
                if find(dWkm(j,7:8)==0)
                    z(j,find(dWkm(j,7:8)==0)) = 1/nnz(dWkm(j,7:8)==0);
                    z(j,find(dWkm(j,7:8)~=0)) = 0;
                end
                
            end
        end
    end
    
    %Update the cluster weights.
    for j=1:k
        distance(j,:,:) = (1-exp((-1.*repmat(landa,N,1)).*((X-repmat(M(j,:),N,1)).^2))).*repmat(v,N,1);
        WBETA = transpose(z(j,:).^beta_z);
        WBETA(WBETA==inf)=0;
        Dw(1,j) = transpose(WBETA) * transpose(reshape(distance(j,:,:),[N,d])) *  transpose(Cluster_elem(j,:).^fuzzy_degree) ;
    end
    
    tmp = sum((repmat(Dw,k,1)./transpose(repmat(Dw,k,1))).^(1/(p-1)));
    tmp(tmp==inf)=0;
    tmp(isnan(tmp))=0;
    W = 1./tmp;
    W(isnan(W))=1;
    W(W==inf)=1;
    
    if nnz(Dw==0)>0
        W(find(Dw==0)) = 1/nnz(Dw==0);
        W(find(Dw~=0)) = 0;
    end
    
    
    %Memory effect.
    W=(1-beta_memory)*W+beta_memory*W_old;
    z=(1-beta_memory)*z+beta_memory*z_old;
    
    Iter=Iter+1;
end

end



