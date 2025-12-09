import math
from typing import Optional, Dict, Any

import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from fpdf import FPDF

# -----------------------------
# NASA POWER API INTEGRATION
# -----------------------------

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
NASA_PARAM = "ALLSKY_SFC_SW_DWN"  # GHI in kWh/m²/day


def get_nasa_ghi_climatology(latitude: float, longitude: float) -> Dict[str, float]:
    """
    Fetch 20-year (2001–2020) solar GHI climatology from NASA POWER API.

    Returns:
        dict: { 'JAN': val, 'FEB': val, ..., 'DEC': val } in kWh/m²/day
    """
    params = {
        "start": 2001,
        "end": 2020,
        "latitude": latitude,
        "longitude": longitude,
        "community": "re",
        "parameters": NASA_PARAM,
        "format": "json",
        "header": "true",
    }

    resp = requests.get(NASA_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    def find_monthly_param(node: Any, param_name: str = NASA_PARAM):
        if isinstance(node, dict):
            # Direct param dict?
            if param_name in node and isinstance(node[param_name], dict):
                maybe = node[param_name]
                month_keys = {
                    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
                }
                if month_keys.issubset(set(maybe.keys())):
                    return maybe

            # Node["parameters"][param_name]
            if "parameters" in node and isinstance(node["parameters"], dict):
                if param_name in node["parameters"]:
                    maybe = node["parameters"][param_name]
                    if isinstance(maybe, dict):
                        month_keys = {
                            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
                        }
                        if month_keys.issubset(set(maybe.keys())):
                            return maybe

            # properties.parameter[param_name]
            if "properties" in node and isinstance(node["properties"], dict):
                props = node["properties"]
                if "parameter" in props and isinstance(props["parameter"], dict):
                    if param_name in props["parameter"]:
                        maybe = props["parameter"][param_name]
                        if isinstance(maybe, dict):
                            month_keys = {
                                "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
                            }
                            if month_keys.issubset(set(maybe.keys())):
                                return maybe

            # Recurse into values
            for v in node.values():
                found = find_monthly_param(v, param_name)
                if found is not None:
                    return found

        elif isinstance(node, list):
            for v in node:
                found = find_monthly_param(v, param_name)
                if found is not None:
                    return found

        return None

    param_block = find_monthly_param(data)
    if param_block is None:
        raise ValueError("Could not find monthly GHI (ALLSKY_SFC_SW_DWN) in NASA response.")

    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    monthly_kwh_per_m2_per_day = {m: float(param_block[m]) for m in months}
    return monthly_kwh_per_m2_per_day


# -----------------------------
# Reverse geocoding for location name
# -----------------------------

def reverse_geocode_city(latitude: float, longitude: float) -> Optional[str]:
    """
    Use OpenStreetMap Nominatim reverse geocoding to get a place name
    (city / town / village / state, plus country) for the given coordinates.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "format": "jsonv2",
            "lat": latitude,
            "lon": longitude,
        }
        headers = {
            "User-Agent": "solar-feasibility-app/1.0 (contact: example@example.com)"
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        address = data.get("address", {}) or {}
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or address.get("state")
        )
        country = address.get("country")

        if city and isinstance(city, str):
            return f"{city}, {country}" if country else city

        return country
    except Exception:
        return None


# -----------------------------
# Core feasibility logic
# -----------------------------

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def build_solar_profile(ghi_daily_kwh_per_m2: Dict[str, float],
                        tilt_gain_factor: float = 1.1) -> pd.DataFrame:
    """
    Convert NASA daily GHI (kWh/m²/day) into a monthly & annual profile.
    """
    rows = []
    annual_ghi_horizontal = 0.0
    annual_ghi_poa = 0.0

    for i, m_short in enumerate(MONTH_NAMES):
        nasa_key = m_short[:3].upper()
        daily_horiz = ghi_daily_kwh_per_m2[nasa_key]
        days = MONTH_DAYS[i]
        monthly_horiz = daily_horiz * days
        monthly_poa = monthly_horiz * tilt_gain_factor

        annual_ghi_horizontal += monthly_horiz
        annual_ghi_poa += monthly_poa

        rows.append({
            "Month": m_short,
            "Days": days,
            "GHI_horizontal_kWh/m²/day": daily_horiz,
            "GHI_horizontal_kWh/m²/month": monthly_horiz,
            "GHI_POA_kWh/m²/month": monthly_poa,
        })

    df = pd.DataFrame(rows)
    df.attrs["annual_ghi_horizontal"] = annual_ghi_horizontal
    df.attrs["annual_ghi_poa"] = annual_ghi_poa
    return df


def classify_solar_resource(annual_poa_kwh_m2: float) -> str:
    if annual_poa_kwh_m2 >= 2200:
        return "Excellent solar resource (top tier for utility-scale PV)."
    elif annual_poa_kwh_m2 >= 1800:
        return "Good solar resource suitable for most PV projects."
    elif annual_poa_kwh_m2 >= 1400:
        return "Moderate solar resource; economics more sensitive to capex and tariff."
    else:
        return "Lower solar resource; PV may still work with strong incentives or high tariffs."


def estimate_pv_yield_and_financials(
    profile_df: pd.DataFrame,
    system_size_kwp: float,
    performance_ratio: float,
    capex_per_kwp: float,
    tariff_per_kwh: float,
    om_percent_of_capex: float,
    project_life_years: int,
    discount_rate_percent: float,
    emission_factor_kg_per_kwh: float,
    capex_subsidy_percent: float,
) -> Dict[str, float]:
    """
    Energy yield, simple payback, LCOE (with & without subsidy),
    capacity factor (as a variable), and GHG savings + equivalents.
    """
    annual_poa_kwh_m2 = profile_df.attrs["annual_ghi_poa"]

    # Energy
    annual_kwh_per_kwp = annual_poa_kwh_m2 * performance_ratio
    annual_energy_kwh = annual_kwh_per_kwp * system_size_kwp

    # Capex & subsidy
    total_capex = capex_per_kwp * system_size_kwp
    subsidy_fraction = capex_subsidy_percent / 100.0
    effective_capex = total_capex * (1.0 - subsidy_fraction)

    # O&M and revenue
    annual_om = total_capex * (om_percent_of_capex / 100.0)
    annual_revenue = annual_energy_kwh * tariff_per_kwh
    annual_net_cashflow = annual_revenue - annual_om

    # Simple payback (using net capex)
    simple_payback_years: Optional[float] = None
    if annual_net_cashflow > 0:
        simple_payback_years = effective_capex / annual_net_cashflow

    # LCOE via capital recovery factor
    r = discount_rate_percent / 100.0
    n = project_life_years
    if r > 0:
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        crf = 1.0 / n

    annualized_capex_no_subsidy = total_capex * crf
    annualized_capex_with_subsidy = effective_capex * crf

    annualized_cost_no_subsidy = annualized_capex_no_subsidy + annual_om
    annualized_cost_with_subsidy = annualized_capex_with_subsidy + annual_om

    lcoe_no_subsidy = (
        annualized_cost_no_subsidy / annual_energy_kwh
        if annual_energy_kwh > 0 else float("nan")
    )
    lcoe_with_subsidy = (
        annualized_cost_with_subsidy / annual_energy_kwh
        if annual_energy_kwh > 0 else float("nan")
    )

    # Capacity factor as a named variable
    capacity_factor = (
        annual_energy_kwh / (system_size_kwp * 8760.0)
        if system_size_kwp > 0 else float("nan")
    )

    # GHG savings (tCO2/year), using effective emission factor
    annual_ghg_savings_tco2 = (
        annual_energy_kwh * emission_factor_kg_per_kwh / 1000.0
        if annual_energy_kwh > 0 else 0.0
    )

    # Equivalents (very approximate)
    # 1 passenger car ~ 4.6 tCO2/year
    # 1 hectare of forest ~ 7 tCO2/year of CO2 uptake
    CAR_TCO2_PER_YEAR = 4.6
    FOREST_TCO2_PER_HA_PER_YEAR = 7.0

    cars_equiv = (
        annual_ghg_savings_tco2 / CAR_TCO2_PER_YEAR
        if CAR_TCO2_PER_YEAR > 0 else 0.0
    )
    forest_ha_equiv = (
        annual_ghg_savings_tco2 / FOREST_TCO2_PER_HA_PER_YEAR
        if FOREST_TCO2_PER_HA_PER_YEAR > 0 else 0.0
    )

    return {
        "annual_kwh_per_kwp": annual_kwh_per_kwp,
        "annual_energy_kwh": annual_energy_kwh,
        "total_capex": total_capex,
        "effective_capex": effective_capex,
        "annual_om": annual_om,
        "annual_revenue": annual_revenue,
        "annual_net_cashflow": annual_net_cashflow,
        "simple_payback_years": simple_payback_years if simple_payback_years is not None else float("nan"),
        "lcoe_no_subsidy": lcoe_no_subsidy,
        "lcoe_with_subsidy": lcoe_with_subsidy,
        "capacity_factor": capacity_factor,
        "annual_ghg_savings_tco2": annual_ghg_savings_tco2,
        "cars_equiv": cars_equiv,
        "forest_ha_equiv": forest_ha_equiv,
        "emission_factor_kg_per_kwh": emission_factor_kg_per_kwh,
    }


# -----------------------------
# PDF report generator
# -----------------------------

def generate_pdf_report(
    location_text: str,
    latitude: float,
    longitude: float,
    system_size_kwp: float,
    performance_ratio: float,
    capex_per_kwp: float,
    tariff_per_kwh: float,
    om_percent_of_capex: float,
    project_life_years: int,
    discount_rate_percent: float,
    capex_subsidy_percent: float,
    displacement_mode: str,
    emission_factor_kg_per_kwh: float,
    annual_horizontal: float,
    annual_poa: float,
    results: Dict[str, float],
    solar_classification: str,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title & location
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Solar PV Feasibility Report", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Location: {location_text}", ln=1)
    pdf.cell(0, 8, f"Coordinates: lat {latitude:.4f}, lon {longitude:.4f}", ln=1)
    pdf.ln(4)

    # Section 1 – Site & resource
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Site & Solar Resource", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Annual GHI (horizontal): {annual_horizontal:,.0f} kWh/m2/year", ln=1)
    pdf.cell(0, 6, f"Annual POA irradiation (tilted): {annual_poa:,.0f} kWh/m2/year", ln=1)
    pdf.multi_cell(0, 6, f"Resource quality: {solar_classification}")
    pdf.ln(2)

    # Section 2 – Assumptions
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. System & Economic Assumptions", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"System size: {system_size_kwp:,.0f} kWp", ln=1)
    pdf.cell(0, 6, f"Performance ratio: {performance_ratio:.2f}", ln=1)
    pdf.cell(0, 6, f"CAPEX: {capex_per_kwp:,.0f} per kWp", ln=1)
    pdf.cell(0, 6, f"Electricity value / tariff: {tariff_per_kwh:.3f} per kWh", ln=1)
    pdf.cell(0, 6, f"Annual O&M: {om_percent_of_capex:.2f}% of CAPEX", ln=1)
    pdf.cell(0, 6, f"Project life: {project_life_years} years", ln=1)
    pdf.cell(0, 6, f"Discount rate: {discount_rate_percent:.1f}%", ln=1)
    pdf.cell(0, 6, f"Capex subsidy / grant: {capex_subsidy_percent:.1f}% of CAPEX", ln=1)
    pdf.cell(
        0,
        6,
        f"Displacement: {displacement_mode} "
        f"(emission factor: {emission_factor_kg_per_kwh:.3f} kg CO2/kWh)",
        ln=1,
    )
    pdf.ln(2)

    # Section 3 – KPIs
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Energy Yield, Financial & GHG KPIs", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Annual yield: {results['annual_kwh_per_kwp']:,.0f} kWh/kWp/year", ln=1)
    pdf.cell(0, 6, f"Annual energy (system): {results['annual_energy_kwh']:,.0f} kWh/year", ln=1)
    pdf.cell(0, 6, f"Capacity factor: {results['capacity_factor']*100:,.1f}%", ln=1)
    pdf.cell(0, 6, f"Total CAPEX (gross): {results['total_capex']:,.0f}", ln=1)
    pdf.cell(0, 6, f"Effective CAPEX (after subsidy): {results['effective_capex']:,.0f}", ln=1)
    pdf.cell(0, 6, f"Annual O&M cost: {results['annual_om']:,.0f}", ln=1)
    pdf.cell(0, 6, f"Annual revenue (energy value): {results['annual_revenue']:,.0f}", ln=1)
    pdf.cell(0, 6, f"Annual net cashflow (before debt): {results['annual_net_cashflow']:,.0f}", ln=1)
    if math.isfinite(results["simple_payback_years"]):
        pdf.cell(0, 6, f"Simple payback (with subsidy): {results['simple_payback_years']:.1f} years", ln=1)
    else:
        pdf.cell(0, 6, "Simple payback: n/a (non-positive net cashflow)", ln=1)
    pdf.cell(
        0,
        6,
        f"Annual GHG savings: {results['annual_ghg_savings_tco2']:,.0f} tCO2/year",
        ln=1,
    )
    pdf.cell(
        0,
        6,
        f"Equivalent to removing ~{results['cars_equiv']:,.0f} passenger vehicles/year",
        ln=1,
    )
    pdf.cell(
        0,
        6,
        f"Or ~{results['forest_ha_equiv']:,.0f} hectares of forest CO2 uptake (approximate)",
        ln=1,
    )
    pdf.ln(2)

    # Section 4 – LCOE
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Levelized Cost of Energy (LCOE)", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"LCOE (no subsidy): {results['lcoe_no_subsidy']:.3f} per kWh", ln=1)
    pdf.cell(0, 6, f"LCOE (with subsidy): {results['lcoe_with_subsidy']:.3f} per kWh", ln=1)
    pdf.multi_cell(
        0,
        6,
        "Note: LCOE is based on capital recovery over the project life, constant annual energy, "
        "and constant O&M. Debt structure and taxes are not included."
    )
    pdf.ln(2)

    # Section 5 – Incentives & grants (conceptual guidance)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "5. Incentives & Grants (conceptual)", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        0,
        6,
        "This tool applies a generic capex subsidy / grant percentage entered by the user. "
        "For a live project, confirm actual incentives available at national and local levels "
        "(for example: renewable energy funds, investment tax credits, import duty waivers, "
        "feed-in tariffs or Contracts-for-Difference, concessional loans, or carbon finance)."
    )
    pdf.ln(2)

    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0,
        5,
        "This is a high-level desktop feasibility view only. Detailed engineering design, "
        "grid studies, land and permitting checks, and full financial modelling are required "
        "before making an investment decision."
    )

    # Safe handling of fpdf2 output (bytes or str)
    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        pdf_bytes = bytes(raw)
    else:
        pdf_bytes = raw.encode("latin1")

    return pdf_bytes


# -----------------------------
# STREAMLIT GUI
# -----------------------------

def main():
    st.set_page_config(page_title="Solar Feasibility (NASA POWER)", layout="wide")

    st.title("🌍 Desktop Solar Feasibility App")
    st.markdown(
        "Uses **NASA POWER** climatology (2001–2020) to estimate solar resource, "
        "PV performance, financials, and GHG emissions savings for any location on Earth."
    )

    col_geo, col_sys = st.columns(2)

    # ---- Location block ----
    with col_geo:
        st.subheader("1️⃣ Location")

        input_mode = st.radio(
            "How do you want to select the location?",
            ["Enter coordinates manually", "Pick on interactive globe/map"],
            index=0,
        )

        latitude: Optional[float] = None
        longitude: Optional[float] = None

        if input_mode == "Enter coordinates manually":
            latitude = st.number_input("Latitude (°)", value=9.0, format="%.6f")
            longitude = st.number_input("Longitude (°)", value=7.0, format="%.6f")
        else:
            st.markdown("Click anywhere on the map to pick a location.")
            m = folium.Map(location=[10, 0], zoom_start=2, tiles="CartoDB positron")
            m.add_child(folium.LatLngPopup())
            map_data = st_folium(m, width=700, height=400)
            if map_data and map_data.get("last_clicked"):
                latitude = map_data["last_clicked"]["lat"]
                longitude = map_data["last_clicked"]["lng"]
                st.success(f"Selected point: lat {latitude:.4f}, lon {longitude:.4f}")
            else:
                st.info("Click on the map to select a location, then press **Run feasibility**.")

        data_source = st.selectbox(
            "Solar data source",
            ["NASA POWER (GHI climatology)"],
            index=0,
        )

        tilt_gain_factor = st.slider(
            "Tilt gain factor (POA vs horizontal GHI)",
            min_value=1.0,
            max_value=1.3,
            value=1.10,
            step=0.02,
            help="Approximate factor to convert horizontal GHI to plane-of-array at optimum tilt.",
        )

    # ---- Economics & GHG block ----
    with col_sys:
        st.subheader("2️⃣ PV System, Economics & GHG")

        system_size_kwp = st.number_input(
            "System size (kWp)",
            min_value=0.1,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
        )
        performance_ratio = st.slider(
            "Performance ratio",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.01,
            help="Typical range 0.75–0.85.",
        )
        capex_per_kwp = st.number_input(
            "CAPEX (per kWp, in your currency)",
            min_value=0.0,
            max_value=10000.0,
            value=800.0,
            step=50.0,
        )
        tariff_per_kwh = st.number_input(
            "Electricity value / tariff (per kWh)",
            min_value=0.0,
            max_value=10.0,
            value=0.15,
            step=0.01,
        )
        om_percent_of_capex = st.number_input(
            "Annual O&M (% of CAPEX)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
        )
        project_life_years = st.number_input(
            "Project life (years)",
            min_value=1,
            max_value=40,
            value=25,
            step=1,
        )
        discount_rate_percent = st.number_input(
            "Discount rate (real, %)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.5,
        )
        capex_subsidy_percent = st.number_input(
            "Capex subsidy / grant (% of CAPEX)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )

        st.markdown("##### Emissions displacement scenario")
        displacement_mode = st.radio(
            "What is the solar generation displacing?",
            ["Grid electricity", "Diesel generators"],
            index=0,
        )

        grid_emission_factor_kg_per_kwh = st.number_input(
            "Grid emission factor (kg CO₂ per kWh displaced)",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.05,
            help="Typical fossil-heavy grids: 0.6–0.9 kgCO₂/kWh. "
                 "Use your regulatory or marginal grid factor if available.",
        )

        diesel_emission_factor_kg_per_kwh = st.number_input(
            "Diesel generator emission factor (kg CO₂ per kWh displaced)",
            min_value=0.0,
            max_value=2.0,
            value=0.8,
            step=0.05,
            help="Approximate diesel genset emissions. Use project-specific data if available.",
        )

    st.markdown("---")

    # ---- Run & display ----
    if st.button("⚡ Run feasibility"):
        if latitude is None or longitude is None:
            st.error("Latitude/longitude could not be determined. Select a point or enter coordinates.")
            return

        try:
            if data_source.startswith("NASA POWER"):
                ghi_daily = get_nasa_ghi_climatology(latitude, longitude)
            else:
                st.error("Only NASA POWER is implemented at the moment.")
                return

            profile_df = build_solar_profile(ghi_daily, tilt_gain_factor=tilt_gain_factor)

            # Choose emission factor based on displacement mode
            if displacement_mode == "Diesel generators":
                emission_factor = diesel_emission_factor_kg_per_kwh
            else:
                emission_factor = grid_emission_factor_kg_per_kwh

            results = estimate_pv_yield_and_financials(
                profile_df=profile_df,
                system_size_kwp=system_size_kwp,
                performance_ratio=performance_ratio,
                capex_per_kwp=capex_per_kwp,
                tariff_per_kwh=tariff_per_kwh,
                om_percent_of_capex=om_percent_of_capex,
                project_life_years=project_life_years,
                discount_rate_percent=discount_rate_percent,
                emission_factor_kg_per_kwh=emission_factor,
                capex_subsidy_percent=capex_subsidy_percent,
            )

            annual_horizontal = profile_df.attrs["annual_ghi_horizontal"]
            annual_poa = profile_df.attrs["annual_ghi_poa"]
            solar_class = classify_solar_resource(annual_poa)

            # Location label
            location_name = reverse_geocode_city(latitude, longitude)
            if location_name:
                location_text = f"{location_name}"
            else:
                location_text = f"lat {latitude:.4f}, lon {longitude:.4f}"

            st.info(f"Location: **{location_text}** (lat {latitude:.4f}, lon {longitude:.4f})")

            # Summary metrics
            st.subheader("3️⃣ Feasibility summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Annual GHI horizontal", f"{annual_horizontal:,.0f} kWh/m²/yr")
                st.metric("Annual GHI POA (tilted)", f"{annual_poa:,.0f} kWh/m²/yr")
            with c2:
                st.metric("Annual yield", f"{results['annual_kwh_per_kwp']:,.0f} kWh/kWp/yr")
                st.metric("Annual energy (system)", f"{results['annual_energy_kwh']:,.0f} kWh/yr")
                st.metric("Capacity factor", f"{results['capacity_factor']*100:,.1f}%")
            with c3:
                st.metric("Total CAPEX (gross)", f"{results['total_capex']:,.0f}")
                st.metric("Effective CAPEX (after subsidy)", f"{results['effective_capex']:,.0f}")
                if math.isfinite(results["simple_payback_years"]):
                    st.metric("Simple payback", f"{results['simple_payback_years']:.1f} years")
                else:
                    st.metric("Simple payback", "n/a")

            # LCOE
            st.subheader("4️⃣ LCOE snapshot")
            st.write(f"- **LCOE (no subsidy):** {results['lcoe_no_subsidy']:.3f} per kWh")
            st.write(f"- **LCOE (with subsidy):** {results['lcoe_with_subsidy']:.3f} per kWh")
            st.caption(
                "LCOE based on capital recovery over project life, constant annual energy and O&M; "
                "debt/taxes not included."
            )

            # GHG impact
            st.subheader("5️⃣ GHG emissions impact")
            st.write(f"- Displacement scenario: **{displacement_mode}**")
            st.write(
                f"- Effective emission factor: {emission_factor:.3f} kg CO₂ per kWh displaced"
            )
            st.write(
                f"- Estimated avoided emissions: "
                f"{results['annual_ghg_savings_tco2']:,.0f} tCO₂ per year"
            )
            st.write(
                f"- Equivalent to removing ~{results['cars_equiv']:,.0f} passenger vehicles per year"
            )
            st.write(
                f"- Or ~{results['forest_ha_equiv']:,.0f} hectares of forest CO₂ uptake (approximate)"
            )

            # Monthly profile (ordered Jan–Dec)
            st.markdown("### 6️⃣ Monthly solar profile")
            profile_df_display = profile_df.copy()

            # Force Month to be an ordered categorical: Jan → Dec
            profile_df_display["Month"] = pd.Categorical(
                profile_df_display["Month"],
                categories=MONTH_NAMES,
                ordered=True,
            )
            profile_df_display = profile_df_display.sort_values("Month")

            profile_df_display["GHI_horizontal_kWh/m²/day"] = profile_df_display[
                "GHI_horizontal_kWh/m²/day"
            ].map(lambda x: round(x, 2))
            profile_df_display["GHI_horizontal_kWh/m²/month"] = profile_df_display[
                "GHI_horizontal_kWh/m²/month"
            ].map(lambda x: round(x, 1))
            profile_df_display["GHI_POA_kWh/m²/month"] = profile_df_display[
                "GHI_POA_kWh/m²/month"
            ].map(lambda x: round(x, 1))

            st.dataframe(profile_df_display, use_container_width=True)
            st.markdown("#### GHI (horizontal) by month – kWh/m²/month")

            profile_df_chart = profile_df.copy()
            profile_df_chart["Month"] = pd.Categorical(
                profile_df_chart["Month"],
                categories=MONTH_NAMES,
                ordered=True,
            )
            profile_df_chart = profile_df_chart.sort_values("Month")

            st.bar_chart(
                profile_df_chart.set_index("Month")["GHI_horizontal_kWh/m²/month"]
            )

            # PDF download
            pdf_bytes = generate_pdf_report(
                location_text=location_text,
                latitude=latitude,
                longitude=longitude,
                system_size_kwp=system_size_kwp,
                performance_ratio=performance_ratio,
                capex_per_kwp=capex_per_kwp,
                tariff_per_kwh=tariff_per_kwh,
                om_percent_of_capex=om_percent_of_capex,
                project_life_years=project_life_years,
                discount_rate_percent=discount_rate_percent,
                capex_subsidy_percent=capex_subsidy_percent,
                displacement_mode=displacement_mode,
                emission_factor_kg_per_kwh=emission_factor,
                annual_horizontal=annual_horizontal,
                annual_poa=annual_poa,
                results=results,
                solar_classification=solar_class,
            )

            st.download_button(
                "⬇️ Download PDF feasibility report",
                data=pdf_bytes,
                file_name="solar_feasibility_report.pdf",
                mime="application/pdf",
            )

        except Exception as e:
            st.error(f"Error while fetching or processing data: {e}")


if __name__ == "__main__":
    main()
