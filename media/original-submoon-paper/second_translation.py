import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 1) Define constants (CGS)
# -------------------------
G = 6.67e-8  # Gravitational constant
yearsec = 3.154e7  # Seconds in a year
T = 4.6e9 * yearsec  # System age (e.g. 4.6 Gyr)
f = 0.4895  # Hill-stability fraction (Domingos+ 2006)
k2p = 0.25  # Tidal Love number of the moon
Qmoon = 100.0  # Tidal dissipation factor of the moon

# -------------------------
# 2) Define planetary data
#    (masses in grams, radii in cm)
# -------------------------
m_earth = 5.974e27
r_earth = 6.378e8

m_jup = 317.8 * m_earth
r_jup = 69911e5

m_sat = 95.16 * m_earth
r_sat = 58232e5

m_ura = 14.54 * m_earth
r_ura = 25362e5

m_nep = 17.15 * m_earth
r_nep = 24622e5

m_k1625 = 4.0 * m_jup  # Example from Teachey & Kipping (2018)
r_k1625 = 11.4 * r_earth

planets = [
    ("Jupiter", m_jup, r_jup),
    ("Saturn", m_sat, r_sat),
    ("Uranus", m_ura, r_ura),
    ("Neptune", m_nep, r_nep),
    ("Earth", m_earth, r_earth),
    ("Kepler1625b", m_k1625, r_k1625)
]


def read_moon_data(filename):
    moon_data = []

    # Planet radii dictionary
    planet_radii = {
        "Earth": 6370,
        "Jupiter": 69911,
        "Saturn": 58232,
        "Neptune": 24622,
        "Uranus": 25362,
        "Kepler1625b": 83893
    }

    with open(filename, "r") as file:
        for line in file:
            # Skip comment lines or empty lines
            if line.startswith(";") or line.strip() == "":
                continue

            parts = line.split()
            if parts[0] == "Host" or parts[0] == "Name":  # Detect and skip the header
                continue

            try:
                host, name, a_km, d_km, mass = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4])
                if host in planet_radii:
                    a_norm = a_km / planet_radii[host]  # Normalize by planet radius
                    # returns host name, moon name, ist distance to host in host radii, the radius in km and mass
                    moon_data.append((host, name, a_norm, d_km/2, mass))
            except ValueError:
                continue  # Skip lines that don't match expected format

    return np.array(moon_data, dtype=object)  # Store as object array to hold mixed data types


moon_data = read_moon_data("satellites_all.txt")


# ----------------------------------------
# 3) Define submoon sizes and compute mass
#    (e.g. 5 km, 10 km, 20 km, etc.)
# ----------------------------------------
rho_submoon = 2.0  # g/cm^3 for submoon
submoon_radii_km = [5, 10, 20, 40, 200]  # example submoon radii in km
submoon_radii = np.array(submoon_radii_km) * 1e5  # convert km to cm
submoon_masses = (4.0 / 3.0) * np.pi * rho_submoon * (submoon_radii ** 3)

print("Submoon masses corresponding to radii (km) ", submoon_radii_km, " : ")
M_lunar = 7.346e22
print([float((4/3 * r**3 * np.pi * rho_submoon)/1e3/M_lunar) for r in submoon_radii], " lunar masses.")

# ----------------------------------------
# 4) Define a range of Rmoon to test
#    and compute Mmoon = 4/3 π ρ R^3
# ----------------------------------------
rho_moon = 2.5  # g/cm^3 for the main moon
Rmoon_array = np.logspace(1e-10, 5, 300)  # ~10 to ~5000 km in cm
Rmoon_array_cm = Rmoon_array * 1e5
Mmoon_array = (4.0 / 3.0) * np.pi * rho_moon * (Rmoon_array_cm ** 3)


# ---------------------------------------------------
# 5) Function to compute critical orbit acrit (in cm)
#    from IDL formula:
#    acrit = (1/f)*[3*Mplan * ( (13/2)*M_sub * (3*k2p*T*Rmoon^5*sqrt(G)/(Mmoon^(8/3)*Qmoon)) )^(6/13)]^(1/3)
# ---------------------------------------------------
def compute_acrit(Mplan, Msub, Rmoon, Mmoon):
    term = ((13.0 / 2.0) * Msub * (3. * k2p * T * Rmoon ** 5 * np.sqrt(G) / (Mmoon ** (8. / 3.) * Qmoon))) ** (6. / 13.)
    return (1.0 / f) * (3. * Mplan * term) ** (1. / 3.)


# ----------------------------------------
# 6) Make the 2×3 figure
# ----------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, (planet_name, Mplan, Rplan) in enumerate(planets):
    ax = axes[i]

    # We only plot up to ~100 planet radii on the x-axis
    max_x = 100.0

    # For each submoon size, compute acrit vs Rmoon_array
    for j, Msub in enumerate(submoon_masses):
        acrit_vals = compute_acrit(Mplan, Msub, Rmoon_array_cm, Mmoon_array)
        # Convert acrit from cm to "planet radii" on the x-axis
        xvals = acrit_vals / Rplan

        # Plot the line
        ax.plot(xvals, Rmoon_array, label=f'R_sub={submoon_radii_km[j]} km')

    # Shading region above the middle line as an example:
    # (You can adapt this logic to replicate IDL's polyfill)
    # Let’s pick the second submoon line to fill above:
    acrit_mid = compute_acrit(Mplan, submoon_masses[1], Rmoon_array_cm, Mmoon_array)
    ax.fill_between(
        acrit_mid / Rplan, Rmoon_array, 1e6,  # fill up to a big number
        color='gray', alpha=0.3
    )

    # Log scale on y
    ax.set_yscale('log')

    ax.set_ylim(10, 5000)  # match IDL range for solar system
    ax.set_xlim(0, max_x)
    ax.set_xlabel(f"Orbital distance ({planet_name} radii)")
    ax.set_ylabel("Radius of moon (km)")
    ax.set_title(planet_name)

    # Optionally plot known satellites:
    # Example: If you have a list of known satellites [a_i, r_i] in the same units,
    # you can do something like:
    # ax.scatter(a_i/Rplan, r_i, c='k', s=20)

    # Add region text
    ax.text(0.05 * max_x, 2000, "Moons of moons stable", color='k')
    ax.text(0.05 * max_x, 20, "No moons of moons", color='k')

    ax.legend(fontsize=8)

    for i in range(len(moon_data)):
        el = moon_data[i]
        if el[0] == planet_name:
            distance = el[2]
            radius = el[3]

            ax.scatter(distance, radius, color="black", s=50, edgecolor="white", label="Real moons")

plt.tight_layout()
plt.show()
