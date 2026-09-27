from wd_spectra.nonlinear import RecoverableEvaluationError


class DenseHeliumDomainError(RecoverableEvaluationError):
    """Reject an out-of-domain trial; never substitute another physical model.

    The nonlinear driver still propagates this error on its initial state.
    Only candidate steps can be rejected and shortened within the SAME EOS.
    """
