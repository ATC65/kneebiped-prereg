# Preregistro — 010_COVERAGE_CONTROLS (corrida v2.4, kb_core 2.3.1-B, kb_cov 1.0)

**Huella del protocolo `kb_cov.COV`: 8bbcfb93da4a2687**
(el cuaderno 010 se detiene si el código ejecutado no tiene esta huella).

Este archivo se versiona en GitHub y se etiqueta `prereg-010` **antes** de ejecutar el cuaderno. Después de etiquetarlo no se modifica:
cualquier análisis añadido más tarde se informa como exploratorio.

## Antecedentes (ya observados antes de fijar la predicción; no forman parte de la prueba)

Todo lo siguiente se conoció antes de redactar la sección de predicción. Por eso, en E2 solo el brazo NN es confirmatorio
y en E3 solo los brazos FT_eq y DAGGER_filt lo son; σ = 0, DART 5 %, BASE, FT y DAGGER son referencias ya vistas.

- **009 (modelos del 009 v5, diagnóstico 009 v6; veredicto APOYADA):** 10/10 BC con σ = 0 tienen el punto fijo de periodo 1 inestable
  con multiplicador dominante real negativo (ρ mediana 2.13, intervalo 1.57–2.40); 0/10 con DART σ = 5 % (ρ mediana 0.52, intervalo 0.47–0.59).
  Fisher p = 1.1 × 10⁻⁵; Wilcoxon sobre ρ p = 0.002. A 300 mapas, σ = 0: 3 P2 y 7 aperiódicas, 0 caídas, 0/10 válidas para el DOE;
  DART 5 %: 9 P1 y 1 P2 (s2024, punto fijo estable con atractor P2 coexistente). Integridad del 009: 10/10 controles.
- **Recálculo independiente (22 nov 2026, `review_checks_offorbit_jacobian.py --floquet`, sin torch, rtol 1e-10):** ρ coincide con el 009
  en los 22 modelos (|Δρ| ≤ 2.8 × 10⁻⁷, mismo signo del multiplicador dominante). Pares por semilla (n = 10), σ = 0 frente a DART 5 %:
  cociente de ganancia transversal respecto del TEC 0.064 frente a 0.116; error relativo del jacobiano transversal 1.003 frente a 0.928;
  RMSE fuera de la órbita 0.787 frente a 0.736 N·m; RMSE sobre la órbita 0.015 frente a 0.019 N·m (Wilcoxon p ≤ 0.006 en todas).
  La correlación ganancia–ρ es fuerte al juntar grupos (Spearman −0.72) pero no dentro de cada grupo (p = 0.14 y 0.16).
- **008 v2 (configuración ff46818298485294; dictamen DAgger NO APOYADO):** con el protocolo 82001/82002 (N = 200/100, H = 20), TEC: S_state 0.440,
  ruido 10 % 0.720. Medianas BASE / FT / DAGGER: S_state 0.673 / 0.730 / 0.478; ruido 10 % 0.900 / 0.925 / 0.570; válidas 10 / 10 / 8
  (DAGGER s13 y s17 con atractor P2). DAGGER − FT en S_state: mediana −0.258, IC95 [−0.288, −0.192], negativo en 10/10;
  ruido 10 %: −0.345, IC95 [−0.405, −0.225]. FT − BASE: +0.045, IC95 [−0.003, 0.073], p = 0.13.
  ρ y supervivencia no se asocian entre las 28 políticas válidas (Spearman 0.09, p = 0.65).
  En la recolección de DAgger, 27/80 rollouts por semilla (mediana) terminan en caída; aportan el 17 % de las filas agregadas y concentran
  las etiquetas saturadas (7.4 % de sus filas frente a 0.1 % en los rollouts sin caída). El monitor de DAgger cae de 0.78 a 0.37 en la
  iteración 1 y sube hasta 0.60 en la 5.
- En los estados de DART 5 %, la etiqueta del punto nominal más cercano difiere de la etiqueta del TEC en 0.146 N·m RMS (propiedad del diseño de E2).
- Revisión del 21 nov 2026: las BC nominales subentrenadas de v2.2.1 tienen un ciclo P1 estable en 8 de 10 semillas.

## Diseño común

- Configuración ganadora del 003 (selected_experiment.json), 10 semillas del 004: 5, 7, 13, 17, 23, 99, 314, 777, 1234, 2024.
- Entrenador: `kb_core.train_bc` (E1). E2 usa `kb_cov.train_bc_tau`, copia literal de `train_bc` con la cota de par fijada a la de la BC DART 5 %
  de la misma semilla; el cuaderno verifica antes de entrenar que sin ese argumento reproduce `train_bc`.
- Diagnóstico idéntico al 009 v6: órbita de 60 y 300 mapas desde el punto fijo del TEC; punto fijo con residuo ≤ 1e-10; multiplicadores con
  h = 1e-4 y sensibilidad con h ∈ {5e-5, 1e-4, 2e-4} (Δρ ≤ 0.01 y mismo veredicto).
- **Variable primaria (E1, E2):** número de semillas con ρ > 1 y multiplicador dominante real negativo.
- Referencias ya calculadas, reutilizadas sin reentrenar: σ = 0 (modelos del 009) y DART 5 % (modelos del 004).

## E2 — Cobertura frente a suavizado

- Brazo NN: estados de `dart_data_set_s05.csv` sin cambios; en los rollouts 1–3, la etiqueta se sustituye por la del punto de
  `expert_data_set.csv` más cercano con la misma fase de rodilla (distancia euclídea en las 6 variables de estado escaladas por la DE del dataset DART).
- **Regla:**
  - COBERTURA ETIQUETADA si NN tiene ≥ 8/10 inestables y Fisher bilateral NN frente a DART 5 % da p < 0.05.
  - REGULARIZACIÓN si NN tiene ≤ 2/10 inestables y Fisher bilateral NN frente a σ = 0 da p < 0.05.
  - MIXTO en cualquier otro caso.

## E1 — Dosis–respuesta en σ

- Niveles nuevos σ = 1, 2 y 3 % de τ_ref, generados con `K.build_dart_dataset` y la semilla DART del 001 (5000); niveles reutilizados 0 y 5 %.
  σ = 10 % solo si se activa `INCLUDE_SIGMA_10` (se declara en el manifiesto).
- λ_flip = multiplicador real negativo más negativo de cada política (NaN si no tiene).
- **Regla:** MECANISMO GRADUAL si la prueba de tendencia de Cochran–Armitage sobre la fracción inestable da p < 0.05 **y**
  Spearman(σ, λ_flip) > 0 con p < 0.05. Se informa σ*, el nivel donde la mediana de λ_flip cruza −1 (interpolación lineal).

## E3 — Controles de DAgger

- FT_eq: mismos datos que FT del 008; épocas por iteración = round(20 × ⌈n_tr,DAgger,i / 32⌉ / ⌈n_tr,FT / 32⌉), para igualar los pasos de gradiente de DAgger.
- DAGGER_filt: igual que DAgger del 008 (mismas semillas de recolección, β, 16 rollouts, 20 épocas), pero solo se agregan los rollouts que no
  terminan en caída.
- Evaluación con el protocolo del 008: S_state en 82001 (N = 200, H = 20); ruido de acción en 82002.
- **Contrastes primarios:** DAGGER − FT_eq y DAGGER_filt − FT_eq. Regla del 008: APOYADO si la mediana pareada > 0, el IC bootstrap del 95 % > 0
  (5000 remuestreos, semilla 84001) y ≥ 80 % de políticas válidas; EFECTO NEGATIVO si todo el IC < 0; NO APOYADO en otro caso.
- Secundarios: FT_eq − FT y DAGGER_filt − DAGGER.

## Secundarias (todos los brazos)

S_state en el conjunto de estabilidad del 006 (semilla 2026, N = 500, H = 30); RMSE fuera de la órbita en 600 estados retenidos (rollouts DART
nuevos con semillas 91050–91053 y 91100–91103); error relativo del jacobiano tangencial y transversal; cociente de ganancia respecto del TEC.
E2: Wilcoxon pareado por semilla NN frente a DART 5 % en esas cuatro métricas, con Holm.

## Integridad (si falla, el dictamen del experimento es NO CONCLUYENTE)

ρ del TEC = 0.4324 (±1e-3); `build_dart_dataset` reproduce `dart_data_set_s05.csv`; `train_bc_tau` ≡ `train_bc`; modelos completos por brazo;
pesos distintos; insensibilidad al paso h; acuerdo con `K.attractor_period`; S_state del TEC igual al del 006; sin HP_OVERRIDE.
E3 además: la evaluación del TEC en 82001 coincide con la del 008.

## Predicción declarada (la completa el autor antes de etiquetar)

E2: COBERTURA ETIQUETADA (NN inestable en ≥ 8/10 y Fisher NN frente a DART 5 % con p < 0.05). Razón: σ = 0 y DART 5 % difieren en la
    ganancia transversal que codifican las etiquetas; NN conserva los estados DART pero elimina esa corrección, de modo que debería comportarse como σ = 0.
E1: MECANISMO GRADUAL, con σ* entre 1 % y 3 %. Razón: la ganancia aumenta con la corrección inyectada; no hay dato previo para σ < 5 %.
E3: DAGGER − FT_eq: EFECTO NEGATIVO (FT ≈ BASE en el 008, así que igualar pasos no debería rescatar a DAgger).
    DAGGER_filt − FT_eq: NO APOYADO o EFECTO NEGATIVO; el filtro quita la mayor parte de las etiquetas saturadas, por lo que se espera
    una recuperación parcial frente a DAGGER (secundario DAGGER_filt − DAGGER > 0), sin superar a FT_eq.

Fecha: 2026-11-23  Autor: Fidel Trejo Orozco
