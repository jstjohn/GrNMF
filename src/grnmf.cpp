// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;




//' A Network Constrained Version of Non-negative Matrix Factorization.
//' 
//' \code{grnmf} is an R/C++ implementation of the algorithm described in "Non-negative Matrix Factorization on Manifold", and modified slightly as 
//'   described in Hofree et al 2013 to put the constraint on features rather than samples.
//' @useDynLib GrNMF
//' @param Xr an p by n non-negative numeric matrix with each column being a sample, and each row corresponding to a feature.
//' @param Wr a symmetric numeric matrix (p by p) with 1 iff x_i,j and x_i',j' are nearest neighbors by some 
//'   graph distance metric, and 0 otherwise. You decide/compute this in advance, 11 is a good neighbor threshold from Hofree et al 2013.
//' @param k the number of inner dimensions (reduced features, or 'rank') to use for the NMF algorithm.
//' @param n_iter the number of optimization loop iterations.
//' @param lambda_multiple the scalar weight applied to the graphical constaint term, or the multiple applied to the update term
//' @param converge threshold for convergence test for early termination. Negative values deactivate this feature. 
//' @param dynamic_lambda Should we use the dynamic lambda updating scheme of Hofree et al? This makes 
//'   lambda_multiple the multiple applied to the dynamic updating term rather than the lambda that is applied to 
//'   directly to the network influence term on the GrNMF objective function.
//' @return a list of \code{U}, 
//'   (sometimes called \code{W}) and \code{V} 
//'   (sometimes called \code{H}) for the NNLS fit.
//'   \code{Max.iter} which stores the maximum iteration before NMF converged 
//'   Additionally the standard NMF objective 
//'   function fit is returned as \code{ObjectiveFitNMF}.
//'   The full GrNMF objective function score is retured as
//'   \code{ObjectiveFitGrNMF}
//' @references Cai, D., He, X., Wu, X., & Han, J. (2008). Non-negative Matrix Factorization on Manifold. 2008 Eighth IEEE International Conference on Data Mining (ICDM), 63–72. doi:10.1109/ICDM.2008.57
//' @references Xu, W., Liu, X., & Gong, Y. (2003). Document clustering based on non-negative matrix factorization. the 26th annual international ACM SIGIR conference (pp. 267–273). New York, New York, USA: ACM. doi:10.1145/860435.860485
//' @references Hofree, M., Shen, J. P., Carter, H., Gross, A., & Ideker, T. (2013). Network-based stratification of tumor mutations. Nature Methods, 10(11), 1108–1115. doi:10.1038/nmeth.2651
//' @export
//' @examples
//' 
//' # The following is adapted from the NMF package vignette
//' # and uses their functions to set up demo data, and as an NMF implementation
//' # comparison (see http://cran.r-project.org/package=NMF)
//' 
//' # First, generate a synthetic dataset with known classes: 100 features, 23 samples (10+5+8)
//' 
//' library(NMF)
//' set.seed(1234)
//' p <- 100; counts <- c(10, 5, 8);
//' n <- sum(counts)
//' x <- syntheticNMF(p, counts)
//' dim(x)
//' # build the true cluster membership
//' groups <- unlist(mapply(rep, seq(counts), counts))
//' # run on a data.frame
//' set.seed(10)
//' system.time(res <- nmf(data.frame(x), 3, nmfAlgorithm("lee"), "random"))
//' 
//' 
//' # Now do the same for GrNMF, and let's compare how close they are
//' d<-as.matrix(dist(x))
//' nn<-5 # 5 nearest neighbors for adjacency graph construction
//' adj <- apply(d, 1, function(drow){
//'   cut <- sort(drow,partial=nn+1)[nn+1] #this is the nn+1 smallest
//'   ifelse(drow<cut,1,0)
//' })
//' adj <- (adj | t(adj)) + 0 #make it symmetric, 
//'   # (the + 0 turns it back to numeric 0/1)
//' sum(adj!=t(adj)) == 0
//' 
//' set.seed(10)
//' 
//' # turn off the graphical part for this comparison
//' # by setting lambda to 0
//' system.time(res2<-grnmf(x,adj,lambda_multiple=0,k=3))
//' 
//' ##
//' # now we can compare the two fits to the NMF
//' # objective function
//' 
//' # first from the NMF package
//' norm(as.matrix(data.frame(x)- (res@@fit@@W %*% res@@fit@@H)),'F')
//' # and next get the NMF objective function fit from grnmf
//' res2$ObjectiveFitNMF
//' 
//' #it should be the same as the GrNMF fit because lambda=0
//' res2$ObjectiveFitNMF == res2$ObjectiveFitGrNMF
//' 
//' # Now test out but use our demo graph
//' set.seed(10)
//' system.time(res3<-grnmf(x, adj, k=3))
//' res3$ObjectiveFitNMF
//' res3$ObjectiveFitGrNMF
//' 
//' 
//' ##
//' # Now do some simple clustering with kmeans
//' 
//' # function to check if our clusters (which are assigned an arbitrary label)
//' # are the same
//' check_fit <- function(cluster){
//'   c1 = cluster[1:counts[1]]
//'   c2 = cluster[(counts[1]+1):(counts[1]+counts[2])]
//'   c3 = cluster[(counts[1]+counts[2]+1):sum(counts)]
//'   c1m = median(c1)
//'   c2m = median(c2)
//'   c3m = median(c3)
//'   if(length(unique(c(c1m,c2m,c3m))) != 3){
//'     cat("Warning, the following error estimate will be",
//'     "incorrect, there are a large number of prediction errors!\n")
//'   }
//'   correct <- ifelse(c(c1-c1m,
//'     c2-c2m,
//'     c3-c3m
//'   ) == 0, 1, 0)
//'   correct
//' }
//' 
//' # on raw X matrix
//' fit_ori <- kmeans(t(x),3)
//' sum(check_fit(fit_ori$cluster))/n
//' 
//' # on standard NMF matrix
//' fit_nmf <- kmeans(t(res@@fit@@H),3)
//' sum(check_fit(fit_nmf$cluster))/n
//' 
//' # our GrNMF should be equivalent when lambda=0
//' fit_nmf2 <- kmeans(res2$V,3)
//' sum(check_fit(fit_nmf2$cluster))/n
//' 
//' # our GrNMF based clusters
//' fit_grnmf <- kmeans(res3$V,3)
//' sum(check_fit(fit_grnmf$cluster))/n
//' 
// [[Rcpp::export]]
List grnmf(NumericMatrix Xr, NumericMatrix Wr, int k=5, int lambda_multiple=1, int n_iter=10000, double converge=1e-6, bool dynamic_lambda = true) {
    // n samples and p variables
    int p = Xr.nrow(), n = Xr.ncol();
    
    if(Wr.nrow() != p || Wr.ncol() != p){
      throw std::invalid_argument("The edge-weight vector W should be p by p (where p are the features, the rows of X)!");
    }
    // k must be no more than the number of features
    if(k > p)
      throw std::invalid_argument("k must be less than or equal to p, the number of columns of Xr");
    
    double lastFit=std::numeric_limits<double>::max();
    
    
    int convergenceCheckFrequency=10;
    int iterations_to_modify_lambda=100;
    double lambda = lambda_multiple;
    if(lambda_multiple<0){
      throw std::invalid_argument("lambda_multiple must be >= 0!");
    }
    
    arma::mat X(Xr.begin(), p, n, false);       // reuses memory and avoids extra copy
    arma::mat W(Wr.begin(), p, p, false);
    arma::mat U(p, k, arma::fill::randu); // initialize U,V to random values in 0-1
    arma::mat V(n, k, arma::fill::randu);
    double minX = X.min(), maxX = X.max();
    
    if(minX < 0){
      throw std::invalid_argument("The input matrix Xr should be non-negative!");
    }
    
    // set U/V to be random values in the range
    // of X, the NMF package in R recommends this
    U*=(maxX-minX);
    U+=minX+1e-4;
    V*=(maxX-minX);
    V+=minX+1e-4;
    
    
    // the D matrix is the diagonal matrix of row (or column since W is symmetric)
    // sums of W. 
    arma::mat D = arma::diagmat(arma::sum(W));
    // L is the graph laplacian matrix
    arma::mat L = arma::mat(D - W);
    
    //loop over iterations
    int it=0;
    for(it = 0; it < n_iter; it++){
      
      if(it % convergenceCheckFrequency == 0){
        double fit =  arma::as_scalar(arma::norm(X-U*V.t(),"fro")) + arma::as_scalar(lambda * arma::trace(U.t() * L * U));
        //allow the user to turn off by setting converge to a negative
        if(std::abs(lastFit-fit) <= converge && converge >= 0)
          break;
        lastFit=fit;
        if(dynamic_lambda && it <= iterations_to_modify_lambda && lambda_multiple>0){
          double raw_fit = arma::as_scalar(arma::norm(X-U*V.t(),"fro"));
          double raw_graph_penalty = std::sqrt(arma::as_scalar(arma::trace(U.t() * L * U)));
          lambda = raw_fit/raw_graph_penalty * lambda_multiple;
        }
      }
      
      //Eq 15 uij update:
      //Eq 15 numerator
      //arma::mat XV=X*V;
      //Eq 15 denominator
      //arma::mat UVtV = U*V.t()*V;
      // carry out Eq 15 update to U
      
      
      // The following does element wize multiplicative updates
      // on the element-wise division, which is equivalent
      // to the following unvectorized loop which is commented out.
      // see http://arma.sourceforge.net/docs.html#operators
      U %= (X * V + lambda * W * U) / (U * V.t() * V + lambda * D * U); // let armadillo try to further optimize this

      
      //Eq 16 vij update:
      //Eq 16 numerator
      //arma::mat XtUpLWV = X.t()*U + lambda*W*V;
      //Eq 17 denominator
      //arma::mat VUtUpLDV = V*U.t()*U+lambda*D*V;
      // carry out Eq 16 update to V
      
      V %= (X.t() * U) / (V * U.t() * U); // let armadillo try to further optimize this
      
      
    } //end iterations
    
    return Rcpp::List::create(
        Rcpp::Named("U") = U,
        Rcpp::Named("V") = V,
        Rcpp::Named("Max.iter") = it,
        Rcpp::Named("ObjectiveFitNMF") = arma::as_scalar(arma::norm(X-U*V.t(),"fro")),
        Rcpp::Named("ObjectiveFitGrNMF") =  arma::as_scalar(arma::norm(X-U*V.t(),"fro")) + arma::as_scalar(lambda * arma::trace(U.t() * L * U))
    ) ;

}
