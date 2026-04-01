def test_typing_exports_expected_aliases():
    from jasmine._typing import Array, OptState, Params, PRNGKey

    assert Array is not None
    assert Params is not None
    assert PRNGKey is not None
    assert OptState is not None
