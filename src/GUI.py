import streamlit as st


class CustomNumeric:
    def variable_increment(
        keys: list[str],
        callback_fns: list[callable],
        values: list[float],
        main_bounds: list[float] = [-1.0, 1.0],
    ):
        """
        Creates a set of numeric fields. The first numeric field controls the increment value of
        the others (the 'main' numeric fields)

        Parameters:
            keys (list[str]) : a list of keys for each of the 'main'
            callback_fns (list[callable]) : a list of callables to use when each main numeric field changes
            values (list[float]) : the starting value of each of the main numeric fields
            main_bounds (list[float]) : the bounds on the main numeric (common to all)

        """

        inc = st.number_input(
            "Step size",
            value=0.01,
            min_value=0.0,
            max_value=0.1,
            key=f"{keys[0]}_increment",
            step=0.005,
            format="%.3f",
        )
        for c_fn, key, v in zip(callback_fns, keys, values):
            st.number_input(
                key,
                value=v,
                min_value=main_bounds[0],
                max_value=main_bounds[1],
                key=key,
                step=inc,
                on_change=c_fn,
                format="%.3f",
            )
