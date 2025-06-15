import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import streamlit.components.v1 as components

st.set_page_config(page_title="Reto F2003B.503 Error404", layout="wide")

# Función para abrir y redimensionar imagen a tamaño fijo (120x120)
def load_and_resize(image_path, size=(120, 120)):
    img = Image.open(image_path)
    # Redimensiona y rellena manteniendo proporción (fit)
    img = img.convert("RGBA")
    img.thumbnail(size, Image.LANCZOS)

    # Crear imagen en blanco (transparente) con tamaño fijo
    background = Image.new("RGBA", size, (255, 255, 255, 0))
    offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
    background.paste(img, offset)
    return background

# Cargar imágenes redimensionadas
img_jd = load_and_resize("johndeere.png")
img_tec = load_and_resize("tec.png")

# CSS personalizado
st.markdown("""
    <style>
        body, .main, .block-container {
            background-color: white !important;
            color: black !important;
        }
        .header-container {
            background-color: #2e7d32;  /* verde */
            padding: 2rem 1rem 1rem 1rem;
            border-bottom: 4px solid #1b5e20;
            border-radius: 0 0 10px 10px;
            text-align: center;
            color: black !important;
        }
        .header-title {
            font-size: 28px !important;
            margin-top: 1rem;
            font-weight: 600;
        }
        div[data-baseweb="tab-list"] {
            background-color: #2e7d32 !important;
            border-radius: 0 0 10px 10px;
            padding-left: 10px;
        }
        button[data-baseweb="tab"] {
            color: white !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #1b5e20 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header completo con fondo verde
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 1])
    with cols[2]:
        img_cols = st.columns(2)
        with img_cols[0]:
            st.image(img_jd)
        with img_cols[1]:
            st.image(img_tec)

    st.markdown('<div class="header-title">Caracterización del sistema de corte en una Rotary Cutter John Deere</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Tabs y contenido
tab1, tab2, tab3 = st.tabs(["🎯 Reto", "💡 Idea", "⚙️ Simulación"])

with tab1:
    st.header("Rotary cutter")
    st.write("Se modeló el sistema de corte en base a diferentes parámetros a considerar proporcionados por la OSF:")
    st.markdown("""
    - Considera potencia máxima de salida del tractor de **50 HP**
    - **540 RPMs** nominales a la salida del tractor
    - Relación de engranaje de la transmisión entrada/salida de **1:1.5**
    - Entre el tractor y la entrada de la transmisión hay un **clutch** que limita el torque a **1000 Nm**
    - Todos los componentes son de **acero**
    """)
    st.write("Como equipo, el objetivo se basó en encontrar parámetros que minimizaran la cantidad de combustible.")
with tab2:
    st.header("Resumen de Reto")
    st.write("Modelación de cortadora con un sistema PD, ajustado manualmente a valores proporcionados por la Organización Socio Formadora, para en base a esto modelar distintos parámetros ajustados por el usuario y que detecte qué configuración le conviene más para el ahorro de combustible.")


with tab3:
    st.header("Simulación")
    st.write("")

    # Crear tres columnas horizontales: parámetros + 2 gráficas
    col1, col2, col3 = st.columns([1, 2, 2])

    # Columna 1: Parámetros
    with col1:
        st.markdown("**Longitud de la barra (metros):**")
        opcion1 = st.selectbox("", [0.4919, 0.9419, 1.0, 1.3919, 1.8419, 2.2919])

        st.markdown("**Densidad del material (kg/m³):**")
        opcion2 = st.selectbox("", [7850, 7975, 7350, 19250, 2700])

        st.markdown("**Radio del disco (metros):**")
        opcion3 = st.selectbox("", [0.2835,0.4085, 0.5335, 0.6585, 0.7835])

        st.markdown('<span style="color:black;"><strong>Posición de las barras respecto al origen (metros):</strong></span>', unsafe_allow_html=True)
        opcion4 = st.number_input(
            label="",
            min_value=0.0,
            max_value=0.7581,
            step=0.001,
            format="%.3f"
        )

    # Columna 2: Área cuchillas + Gráfica 1
    with col2:
        st.markdown("### Gráfica 1")

        # --- Simulación (usa los valores elegidos arriba) ---
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.integrate import solve_ivp

        # Parámetros físicos ajustables
        rho_pasto = 2614  # plantas/m^2
        r_plato = opcion3
        r_union = opcion4

        if r_union > r_plato - 0.0254:
            r_union = r_plato - 0.0254

        L = opcion1
        rho = opcion2
        rho_barra = rho
        thickness = 0.013
        width = 0.076
        A_barra = thickness * width
        m_barra = rho_barra * A_barra * L

        I_barra = (1 / 3) * m_barra * L * 2 + m_barra * r_union ** 2

        A_plato = np.pi * r_plato ** 2
        rho_plato = rho
        espesor = 0.007
        I_plato = (1 / 2) * A_plato * espesor * rho_plato * r_plato ** 2

        I_total = 2 * I_barra + I_plato

        k = 0.0013
        A = 1.0
        B = 1.0
        C = 1.0

        def F_pasto(t): return 9.25 * np.sin(t) + 26.65
        def dF_pasto_dt(t): return 9.25 * np.cos(t)

        def model(t, omega):
            F = F_pasto(t)
            dF = dF_pasto_dt(t)
            R_eff = L * (1 - np.exp(-k * omega[0]))
            tau_motor = A * F + B * dF + C
            tau_pasto = 2 * rho_pasto * F * (R_eff ** 3)
            domega_dt = (tau_motor - tau_pasto) / I_total
            return [domega_dt]

        omega0 = [0.0]
        t_span = (0, 150)
        t_eval = np.linspace(*t_span, 400)

        sol = solve_ivp(model, t_span, omega0, t_eval=t_eval, method='RK45')
        t = sol.t
        omega = sol.y[0]

        F_vals = F_pasto(t)
        dF_vals = dF_pasto_dt(t)
        R_eff_vals = L * (1 - np.exp(-k * omega))

        tau_motor_vals = A * F_vals + B * dF_vals + C
        tau_pasto_vals = 2 * rho_pasto * F_vals * (R_eff_vals ** 3)

        # --- Gráfico usando matplotlib ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, omega, label=r'$\omega(t)$ [rad/s]')
        ax.set_title("Velocidad angular del plato con dos cuchillas")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Velocidad angular [rad/s]")
        ax.grid(True)
        ax.legend()

        # Mostrar figura en Streamlit
        st.pyplot(fig)


        # Columna 3: Combustible acumulado + Gráfica 2
        with col3:

            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.integrate import solve_ivp

            # Parámetros físicos ajustables
            rho_pasto = 2614  # plantas/m^2
            r_plato = opcion3
            r_union = opcion4

            if r_union > r_plato - 0.0254:
                r_union = r_plato - 0.0254

            L = opcion1
            rho = opcion2
            rho_barra = rho
            thickness = 0.013
            width = 0.076
            A_barra = thickness * width
            m_barra = rho_barra * A_barra * L

            I_barra = (1 / 3) * m_barra * L * 2 + m_barra * r_union ** 2

            A_plato = np.pi * r_plato ** 2
            rho_plato = rho
            espesor = 0.007
            I_plato = (1 / 2) * A_plato * espesor * rho_plato * r_plato ** 2

            I_total = 2 * I_barra + I_plato

            k = 0.0013
            A = 1.0
            B = 1.0
            C = 1.0

            def F_pasto(t): return 9.25 * np.sin(t) + 26.65
            def dF_pasto_dt(t): return 9.25 * np.cos(t)

            def model(t, omega):
                F = F_pasto(t)
                dF = dF_pasto_dt(t)
                R_eff = L * (1 - np.exp(-k * omega[0]))
                tau_motor = A * F + B * dF + C
                tau_pasto = 2 * rho_pasto * F * (R_eff ** 3)
                domega_dt = (tau_motor - tau_pasto) / I_total
                return [domega_dt]

            omega0 = [0.0]
            t_span = (0, 150)
            t_eval = np.linspace(*t_span, 400)

            sol = solve_ivp(model, t_span, omega0, t_eval=t_eval, method='RK45')
            t = sol.t
            omega = sol.y[0]
            F_vals = F_pasto(t)
            dF_vals = dF_pasto_dt(t)
            R_eff_vals = L * (1 - np.exp(-k * omega))

            tau_motor_vals = A * F_vals + B * dF_vals + C
            tau_pasto_vals = 2 * rho_pasto * F_vals * (R_eff_vals ** 3)
            F_vals = F_pasto(sol.t)

            P_corte = (tau_motor_vals) * omega / 1000  # (Nm * rad/s) / 1000 = kW

            #Q_full = 0.4115 * P_corte+5
            R_ec = (P_corte) / 43

            Q_s = (2.64 * R_ec) + 3.91 - (0.203) * np.sqrt((738 * R_ec) + 173)

            Q_full = Q_s * (P_corte)+3

            Ppto = 37.3          # kW, potencia nominal PTO (50 hp ≈ 37.3 kW)

            total_consumo = np.trapz(Q_full, t)/(1)
            total_consumo = np.round(total_consumo, 5)  # Redondear a 2 decimales
            #st.metric(label="", value="{:.2f} L".format(total_consumo))

            st.markdown("### Gráfica 2")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(t, Q_full, label='Consumo full-throttle [L/h]')
            ax2.set_title("Consumo instantáneo de combustible")
            ax2.set_xlabel("Tiempo [s]")
            ax2.set_ylabel("Razón de Consumo instantáneo [L/h]")
            ax2.grid(True)
            ax2.legend()
            st.pyplot(fig2)

            Q_full_Lps = Q_full / 3600.0

            # 2) Calcular diferencial de tiempo
            dt = np.diff(t)

            # 3) Inicializar vector de consumo acumulado
            consumo_acum = np.zeros_like(t)

            # 4) Integrar por rectángulos: consumo_acum[i] = ∑ Q_full_Lps[j]*dt[j]
            consumo_acum[1:] = np.cumsum(Q_full_Lps[:-1] * dt)
            st.markdown("### Consumo acumulado de combustible en 150 segundos")
            st.markdown(
                f'<span style="color:black; font-size: 1.5em;"><strong>{np.round(consumo_acum[-1],5):.5f} L</strong></span>',
                unsafe_allow_html=True
            )
        # Nueva fila para gráfica 3, centrada ocupando col2 + col3
        _, col_center, _ = st.columns([1, 4, 1])
        with col_center:
            st.markdown("### Gráfica 3: Animación del corte de pasto")
            st.image("animacion_cortadora.gif", caption="Animación del corte de pasto", use_container_width=True)
