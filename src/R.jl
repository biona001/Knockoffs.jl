# Zihuai's R code for solving ME problem using PCA
# must install the following R packages: RSpectra, Matrix
function solve_group_max_entropy_pca_zihuai(
    Sigma::AbstractMatrix{T}, # covariance matrix
    M::Int, # number of knockoffs
    clusters::AbstractVector # group membership
    ) where T

    # pass variables from Julia to R
    @rput Sigma M clusters

    # Zihuai's PCA function: basically, for each group, compute PCA
    # of the group, obtaining eigenvectors v1,...,vg. Then define 
    # Sg = s1*v1 + ... + sg*vg where we optimize only s1,...,sg
    R"""
    library("Matrix")
    create.solve_group_ME_pca_M <- function(Sigma, M=1, clusters, ini.S=NULL, pc.max=Inf, rep.max=Inf, rep.method="R2", gaptol=1e-6, maxit=1000, verbose=FALSE) {
        # Check that covariance matrix is symmetric
        stopifnot(isSymmetric(Sigma))
        # Convert the covariance matrix to a correlation matrix
        G = cov2cor(Sigma)
        p = dim(G)[1]
        
        # Check that the input matrix is positive-definite
        if (!is_posdef(G)) {
            warning('The covariance matrix is not positive-definite: knockoffs may not have power.', immediate.=T)
        }
        
        #clustering to identify tightly linked variants
        clusters.index<-match(unique(clusters),clusters)
        p.clusters<-max(clusters)
        n.clusters <- table(clusters) #reweight clusters
        
        # prepare predefined matrices
        V.pc<-c();V.rep<-c()
        for (j in 1:p.clusters){
            block.index<-which(clusters==j)
            block.G<-G[block.index,block.index,drop=F]
            #PCA
            if(pc.max>0){
                block.fit<-eigen(block.G)
                v<-matrix(0,p,min(n.clusters[j],pc.max))
                v[block.index,]<-block.fit$vectors[,1:min(n.clusters[j],pc.max)]
                V.pc<-cbind(V.pc,v)
            }
            if(rep.max>0){
                #ID
                if(rep.method=='ID'){
                    A<-chol(block.G)
                    block.fit<-rid(A,ncol(A),k=min(n.clusters[j],rep.max),rand=F,idx_only=T)
                    v<-matrix(0,p,min(n.clusters[j],rep.max))
                    v[block.index,]<-diag(1,nrow(block.G))[,block.fit$idx]
                    V.rep<-cbind(V.rep,v)
                }
                #SubsetC
                if(rep.method=='R2'){
                    block.fit<-subsetC(block.G,k=min(n.clusters[j],rep.max))
                    v<-matrix(0,p,min(n.clusters[j],rep.max))
                    v[block.index,]<-diag(1,nrow(block.G))[,block.fit$indices]
                    V.rep<-cbind(V.rep,v)
                }
            }
        }
        V<-Matrix(cbind(V.pc,V.rep))
        temp<-apply(V,2,paste,collapse="")
        V<-V[,match(unique(temp),temp)]
        
        # initial matrix
        #alpha<-1e-8
        #Sigma<-(1-alpha)*Sigma+diag(alpha,p)
        eigen.fit<-eigen(Sigma)
        if(length(ini.S)!=0){S<-ini.S}else{
        #inv.Sigma<-solve(Sigma)
        #S<-diag(1/irlba(inv.Sigma,nv=1)$d,p)
        S<-diag(min(eigen.fit$values),p)
        #S<-diag(1e-8,p)
        }
        S<-Matrix(S)
        E<-Matrix((M+1)/M*Sigma-S)
        inv.E<-eigen.fit$vectors%*%as.matrix(1/((M+1)/M*eigen.fit$values-min(eigen.fit$values))*t(eigen.fit$vectors))#solve(E)
        inv.S<-Matrix(diag(1/S[1,1],p))#Matrix(diag(1/min(eigen.fit$values),p))
        obj<-sum(log((M+1)/M*eigen.fit$values-min(eigen.fit$values)))+M*p*log(min(eigen.fit$values))#log(det(E))+M*log(det(S))
        #obj<-0
        print(obj)
        
        new.obj<-obj
        for(i in 1:maxit){
            change_obj<-0
            for (l in 1:ncol(V)){
                v<-V[,l,drop=F]
                alpha_S<-as.numeric(t(v)%*%inv.S%*%v)
                alpha_E<-as.numeric(t(v)%*%inv.E%*%v)
                delta_l<-max(min((M*alpha_S-alpha_E)/(M+1)/alpha_S/alpha_E,1/alpha_E),-1/alpha_S)
                S<-S+delta_l*v%*%t(v)
                
                change_obj_l<-log(1-delta_l*alpha_E+1e-8)+M*log(1+delta_l*alpha_S+1e-8)
                change_obj<-change_obj+change_obj_l
                
                temp<-t(v)%*%inv.E
                inv.E<-inv.E-t(temp)%*%((-delta_l^(-1)+alpha_E)^(-1)*temp)
                temp<-t(v)%*%inv.S
                inv.S<-inv.S-t(temp)%*%((delta_l^(-1)+alpha_S)^(-1)*temp)
            }
            new.obj<-new.obj+change_obj
            if(change_obj/abs(new.obj)<=0.001){break}
            print(new.obj)
        }
        
        # Compensate for numerical errors (feasibility)
        psd = 0;
        s_eps = 1e-8;
        while (psd==0) {
            psd = is_posdef((M+1)/M*G-S*(1-s_eps)) #change 2 to (M+1)/Ms
            if (!psd) {
                s_eps = s_eps*10
            }
        }
        S = S*(1-s_eps)
        S<-Matrix(S)
        
        # Scale back the results for a covariance matrix
        return(list(S=diag(Sigma)^(1/2)*t(diag(Sigma)^(1/2)*S),clusters=clusters,obj.tr=sum(diag(S)),obj.me=new.obj))
    }
    is_posdef = function(A, tol=1e-9) {
        p = nrow(matrix(A))
        
        if (p<500) {
          lambda_min = min(eigen(A)$values)
        }
        else {
          oldw <- getOption("warn")
          options(warn = -1)
          lambda_min = RSpectra::eigs(A, 1, which="SM", opts=list(retvec = FALSE, maxitr=100, tol))$values
          options(warn = oldw)
          if( length(lambda_min)==0 ) {
            # RSpectra::eigs did not converge. Using eigen instead."
            lambda_min = min(eigen(A)$values)
          }
        }
        return (lambda_min>tol*10)
    }
    subsetC <- function(C, k=NA, traceit=FALSE){
        ## C correlation matrix
        ## k subset size
        do.adaptive <- is.na(k)
        p <- ncol(C)
        if (do.adaptive) {
          k <- p-1
        }
        indices <- rep(0, k)
        RSS0 <- p
        R2 <- double(k)
        vlist = seq(p)
        for(i in 1:k){
          fit1 <- step1(C, RSS0=RSS0, vlist=vlist)
          indices[i] <- fit1$index
          C <- as.matrix(fit1$C)
          vlist <- fit1$vlist
          R2[i] <- fit1$R2
          if(traceit)cat(i, "index", fit1$index, "Variance Explained", fit1$variance,"R-squared",fit1$R2,"\n")
          
          # if there is at least 3 R2 values,
          # check early stopping rule
          if (do.adaptive && (i >= 3)) {
            rsq_u <- R2[i]
            rsq_m <- R2[i-1]
            rsq_l <- R2[i-2]
            if (check_early_stopping_rule(rsq_l, rsq_m, rsq_u)) {
              indices <- indices[1:i]
              R2 <- R2[1:i]
              break
            }
          }
        }
        list(indices = indices, R2=R2)
    }  
    step1 <- function(C,vlist=seq(ncol(C)),RSS0=sum(diag(C)),zero=1e-12){
        dC <- diag(C)
        rs <- colSums(C^2)/dC
        imax <- order(rs,decreasing=TRUE)[1]
        vmin <- sum(dC) - rs[imax]
        residC = C - outer(C[,imax],C[,imax],"*")/C[imax,imax]
        index = vlist[imax]
        izero = diag(residC) <= zero
        list(index = index, variance = vmin, R2 = 1-vmin/RSS0, C=residC[!izero,!izero],vlist=vlist[!izero])
    }

    # solve ME-PCA problem
    result <- create.solve_group_ME_pca_M(Sigma, M, clusters)
    S <- as.matrix(result$S)
    obj <- result$obj.me
    """

    # pass R result back to Julia
    @rget S obj
    return S, Float64[], obj
end