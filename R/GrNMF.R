#' R Functions Related to Network Constrained NMF
#'
#' The GrNMF package implements the algorithm described in Cai et al's
#' "Non-negative Matrix Factorization on Manifold". Rather than applying the graph 
#' constraint to samples though, we apply it to features, as in Hofree et al. 2013.
#' Also we implement other heuristics from Hofree's method for fitting
#' this network constrained NMF, namely a dynamically updating the lambda multiple
#' on the influence of the network term in the fitting procedure.
#'
#' The \pkg{GrNMF} currently contains one function which 
#' implements this method
#' \code{\link{grnmf}}.
#' @references Cai, D., He, X., Wu, X., & Han, J. (2008). Non-negative Matrix Factorization on Manifold. 2008 Eighth IEEE International Conference on Data Mining (ICDM), 63-72. doi:10.1109/ICDM.2008.57
#' @references Xu, W., Liu, X., & Gong, Y. (2003). Document clustering based on non-negative matrix factorization. the 26th annual international ACM SIGIR conference (pp. 267-273). New York, New York, USA: ACM. doi:10.1145/860435.860485
#' @references Hofree, M., Shen, J. P., Carter, H., Gross, A., & Ideker, T. (2013). Network-based stratification of tumor mutations. Nature Methods, 10(11), 1108-1115. doi:10.1038/nmeth.2651
#' @docType package
#' @name GrNMF
NULL