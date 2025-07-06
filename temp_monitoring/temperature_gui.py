import streamlit as st
from asgard_alignment.controllino import Controllino

st.title("Controllino Control Panel")

page = st.selectbox("Select Page", ["Fan Control", "PI Loop Selection"])

# Connection settings
ip = "192.168.100.10"
port = 23

if "controllino" not in st.session_state:
    try:
        st.session_state["controllino"] = Controllino(ip, port, init_motors=False)
        st.session_state["ip"] = ip
        st.session_state["port"] = port
        st.success("Connected to Controllino.")
    except Exception as e:
        st.error(f"Failed to connect: {e}")

controllino = st.session_state.get("controllino", None)

if controllino:
    if page == "Fan Control":
        st.header("Fan Modulation")

        lower_fan_value = st.number_input(
            "Lower Fan Value", min_value=10, max_value=255, value=128, step=1
        )
        if st.button("Set Lower Fan"):
            try:
                controllino.modulate("Lower Fan", lower_fan_value)
                st.success(f"Lower Fan set to {lower_fan_value}")
            except Exception as e:
                st.error(f"Failed to set Lower Fan: {e}")

        upper_fan_value = st.number_input(
            "Upper Fan Value", min_value=10, max_value=255, value=128, step=1
        )
        if st.button("Set Upper Fan"):
            try:
                controllino.modulate("Upper Fan", upper_fan_value)
                st.success(f"Upper Fan set to {upper_fan_value}")
            except Exception as e:
                st.error(f"Failed to set Upper Fan: {e}")

    elif page == "PI Loop Selection":
        st.header("PI Loop Selection")

        for loop_name in ["Lower", "Upper"]:
            st.subheader(f"{loop_name} PI Loop")
            # Try to get current info for this loop
            info = None
            try:
                info = controllino.read_PI_loop_info(loop_name)
            except Exception as e:
                st.error(f"Failed to read {loop_name} PI loop info: {e}")

            if info:
                setpoint = st.number_input(
                    f"{loop_name} Setpoint",
                    value=info["setpoint"],
                    key=f"{loop_name}_setpoint",
                    step=1,
                )
                k_prop = st.number_input(
                    f"{loop_name} k_prop",
                    value=int(info["k_prop"]),
                    key=f"{loop_name}_k_prop",
                    step=1,
                )
                k_int = st.number_input(
                    f"{loop_name} k_int",
                    value=int(info["k_int"]),
                    key=f"{loop_name}_k_int",
                    step=1,
                )
            else:
                setpoint = st.number_input(
                    f"{loop_name} Setpoint", key=f"{loop_name}_setpoint", step=1
                )
                k_prop = st.number_input(
                    f"{loop_name} k_prop", key=f"{loop_name}_k_prop", step=1
                )
                k_int = st.number_input(
                    f"{loop_name} k_int", key=f"{loop_name}_k_int", step=1
                )

            if st.button(f"Set {loop_name} PI Loop", key=f"{loop_name}_set_btn"):
                try:
                    controllino.set_PI_loop(
                        loop_name, int(setpoint), int(k_prop), int(k_int)
                    )
                    st.success(f"{loop_name} PI loop parameters updated.")
                except Exception as e:
                    st.error(f"Failed to set {loop_name} PI loop parameters: {e}")
