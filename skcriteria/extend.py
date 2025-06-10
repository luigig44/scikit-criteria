#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: BSD-3 (https://tldrlegal.com/license/bsd-3-clause-license-(revised))
# Copyright (c) 2016-2021, Cabral, Juan; Luczywo, Nadia
# Copyright (c) 2022-2025 QuatroPe
# All rights reserved.

# =============================================================================
# DOCS
# =============================================================================

"""Functionalities for the user's extension of scikit-criteria.

This module introduces decorators that enable the creation of aggregation and
transformation models using only functions.

It is important to note that models created with these decorators are much less
flexible than those created using inheritance and lack certain properties of
real objects.

"""

# =============================================================================
# IMPORTS
# =============================================================================ç

from .utils import hidden

with hidden():
    from functools import partial
    import inspect
    import warnings

    from .core import SKCMethodABC
    from .agg import RankResult, SKCDecisionMakerABC
    from .preprocessing import SKCTransformerABC
    from .utils import doc_inherit

# =============================================================================
# CONSTANTS
# =============================================================================

_MODEL_PARAMETERS = (
    "matrix",
    "objectives",
    "weights",
    "dtypes",
    "alternatives",
    "criteria",
    "hparams",
)


# =============================================================================
# WARNINGS
# =============================================================================


class NonStandardNameWarning(UserWarning):
    """Custom warning class to indicate that a name does not follow a \
    specific standard.

    This warning is raised when a given name does not adhere to the specified
    naming convention.

    """


warnings.simplefilter("always", NonStandardNameWarning)

# =============================================================================
# LOGIC
# =============================================================================


def _create_init_signature(kwargs):
    """Create an initialization signature for a class based on the provided \
    keyword arguments."""
    params = [
        inspect.Parameter(
            name, default=default, kind=inspect.Parameter.KEYWORD_ONLY
        )
        for name, default in kwargs.items()
    ]
    signature = inspect.Signature(params)

    return signature


def _check_model_CapWords_convention_name(model_name):
    """Check if a model name follows the 'CapWords' convention and issue a \
    warning if not."""
    if model_name[0].islower():
        msg = (
            "Models names should normally use the 'CapWords' convention."
            f"try change {model_name!r} to "
            f"{model_name.title()!r} or {model_name.upper()!r}"
        )
        warnings.warn(msg, category=NonStandardNameWarning, stacklevel=3)


def _check_function_parameters(func):
    """Raise a type error if a function has invalid parameters according to a \
    predefined set."""
    has_kwars = False
    not_used = set(_MODEL_PARAMETERS)
    for pname, param in inspect.signature(func).parameters.items():
        if param.kind is param.VAR_KEYWORD:
            has_kwars = True
            continue
        elif pname not in _MODEL_PARAMETERS:
            raise TypeError(
                f"{func.__qualname__}() has an invalid parameter {pname!r}"
            )
        not_used.remove(pname)
    if not_used and not has_kwars:
        raise TypeError(
            f"{func.__qualname__}() is missing the parameter(s) {not_used}"
        )


class _AutoMethodMixin(SKCMethodABC):
    _skcriteria_abstract_class = True

    def __init_subclass__(cls, from_=None, hparams={}, **kwargs):
        if from_ is not None:
            _check_model_CapWords_convention_name(from_.__name__)
            _check_function_parameters(from_)
            cls._skcriteria_parameters = frozenset(hparams)
            cls.__init__.__signature__ = _create_init_signature(hparams)
            cls.__doc__ = from_.__doc__
            cls.__name__ = from_.__name__
            cls.__qualname__ = from_.__qualname__
            cls.__module__ = from_.__module__
            cls._skcriteria_auto_function = staticmethod(from_)
        else:
            cls._skcriteria_abstract_class = True
        super().__init_subclass__(**kwargs)

    def __init__(self, **kwargs):
        try:
            bound = self.__init__.__signature__.bind(**kwargs)
        except TypeError as err:
            raise TypeError(f"{type(self).__name__}.__init__() {err}")
        bound.apply_defaults()
        self.__dict__.update(bound.kwargs)


def _mkmaker(cls):
    @doc_inherit(cls)
    def mk(maybe_func=None, **hparams):
        if maybe_func is None:
            return partial(mk, **hparams)

        class subcls(cls, from_=maybe_func, hparams=hparams):
            pass

        return subcls

    return mk


@_mkmaker
class mkagg(SKCDecisionMakerABC, _AutoMethodMixin):
    """Decorator factory function for creating aggregation classes.

    Parameters
    ----------
    maybe_func : callable, optional
        Optional aggregation function to be wrapped into a class. If provided,
        the decorator is applied immediately.

        The decorated function should receive the parameters 'matrix',
        'objectives', 'weights', 'dtypes', 'alternatives', 'criteria',
        'hparams', or kwargs.

        Additionally, it should return an array with rankings for each
        alternative and an optional dictionary with calculations that you wish
        to store in the 'extra' attribute of the ranking."

    **hparams : keyword arguments
        Hyperparameters specific to the aggregation function.

    Returns
    -------
    Agg : class or decorator
        Aggregation class decorator or Aggregatio model with added
        functionality.

    Notes
    -----
    This decorator is designed for creating aggregation model from aggregation
    functions. It provides an interface for creating aggregated
    decision-making models.

    Examples
    --------
    >>> @mkagg
    >>> def MyAgg(**kwargs):
    >>>     # Implementation of the aggregation function

    The above example will create an aggregation class with the name 'MyAgg'
    based on the provided aggregation function.

    >>> @mkagg(foo=1)
    >>> def MyAgg(**kwargs):
    >>>     # Implementation of the aggregation function

    The above example will create an aggregation class with the specified
    hyperparameter 'foo' and the name 'MyAgg'.

    """

    @doc_inherit(SKCDecisionMakerABC._evaluate_data)
    def _evaluate_data(self, **kwargs):
        rank, extra = self._skcriteria_auto_function(hparams=self, **kwargs)
        return rank, extra

    @doc_inherit(SKCDecisionMakerABC._make_result)
    def _make_result(self, alternatives, values, extra):
        return RankResult(
            self.get_method_name(),
            alternatives=alternatives,
            values=values,
            extra=extra,
        )


@_mkmaker
class mktransformer(SKCTransformerABC, _AutoMethodMixin):
    """Decorator factory function for creating transformation classes.

    Parameters
    ----------
    maybe_func : callable, optional
        Optional transformation function to be wrapped into a class. If
        provided, the decorator is applied immediately.

        The decorated function should receive the parameters 'matrix',
        'objectives', 'weights', 'dtypes', 'alternatives', 'criteria',
        'hparams', or kwargs.

        In addition, it must return a dictionary whose keys are some
        as the received parameters (including the keys in 'kwargs').
        These values replace those of the original array.
        If you return 'hparams,' the transformer will ignore it.

        If you want the transformer to infer the types again, return
        `dtypes` with value `None`.

        It is the function's responsibility to maintain compatibility.

    **hparams : keyword arguments
        Hyperparameters specific to the transformation function.

    Returns
    -------
    Trans : class or decorator
        Transformation class decorator or Transformation model with added
        functionality.

    Notes
    -----
    This decorator is designed for creating transformation models from
    transformation functions. It provides an interface for creating transformed
    decision-making models.

    Examples
    --------
    >>> @mktrans
    >>> def MyTrans(**kwargs):
    >>>     # Implementation of the transformation function
    >>>     pass

    The above example will create a transformation class with the name
    'MyTrans' based on the provided transformation function.

    >>> @mktrans(foo=1)
    >>> def MyTrans(**kwargs):
    >>>     # Implementation of the transformation function
    >>>     pass

    The above example will create a transformation class with the specified
    hyperparameter 'foo' and the name 'MyTrans'.
    """

    @doc_inherit(SKCTransformerABC._transform_data)
    def _transform_data(self, **kwargs):
        tdata = self._skcriteria_auto_function(hparams=self, **kwargs)
        # if the function return tdata we will remove it
        tdata.pop("hparams", None)
        # replace the old values with the new ones
        kwargs.update(tdata)
        return kwargs
