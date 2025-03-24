#version 460 core

uniform float now = 0.0;
uniform float delta = 0.8;
uniform float char_length = 1.0;

in float time;
in float arc;
in float v;
in float varc;
in float sarc;
in float speed;
flat in uint stroke;

layout (location = 0) out vec4 frag_color;

// float colormap_red(float x) {
//     return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
// }

// float colormap_green(float x) {
//     return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
// }

// vec4 colormap(float x) {
//     float r = clamp(colormap_red(x), 0.0, 1.0);
//     float g = clamp(colormap_green(x), 0.0, 1.0);
//     float b = 1.0;
//     return vec4(r, g, b, 1.0);
// }

// vec4 colormap(float x) {
//     float v = cos(133.0 * x) * 28.0 + 230.0 * x + 27.0;
//     if (v > 255.0) {
//         v = 510.0 - v;
//     }
//     v = v / 255.0;
//     return vec4(v, v, v, 1.0);
// }

// float colormap_f(float x) {
//   if (x < 0.8110263645648956) {
//     return (((4.41347880412638E+03 * x - 1.18250308887283E+04) * x + 1.13092070303101E+04) * x - 4.94879610401395E+03) * x + 1.10376673162241E+03;
//   } else {
//     return (4.44045986053970E+02 * x - 1.34196160353499E+03) * x + 9.26518306556645E+02;
//   }
// }

// float colormap_red(float x) {
//   if (x < 0.09384074807167053) {
//     return 7.56664615384615E+02 * x + 1.05870769230769E+02;
//   } else if (x < 0.3011957705020905) {
//     return (-2.97052932130813E+02 * x + 4.43575866219751E+02) * x + 1.37867123966178E+02;
//   } else if (x < 0.3963058760920129) {
//     return 8.61868131868288E+01 * x + 2.18562881562874E+02;
//   } else if (x < 0.5) {
//     return 2.19915384615048E+01 * x + 2.44003846153861E+02;
//   } else {
//     return colormap_f(x);
//   }
// }

// float colormap_green(float x) {
//   if (x < 0.09568486400411116) {
//     return 2.40631111111111E+02 * x + 1.26495726495727E+00;
//   } else if (x < 0.2945883673263987) {
//     return 7.00971783488427E+02 * x - 4.27826773670273E+01;
//   } else if (x < 0.3971604611945229) {
//     return 5.31775726495706E+02 * x + 7.06051282052287E+00;
//   } else if (x < 0.5) {
//     return 3.64925470085438E+02 * x + 7.33268376068493E+01;
//   } else {
//     return colormap_f(x);
//   }
// }

// float colormap_blue(float x) {
//   if (x < 0.09892375498249567) {
//     return 1.30670329670329E+02 * x + 3.12116402116402E+01;
//   } else if (x < 0.1985468629735229) {
//     return 3.33268034188035E+02 * x + 1.11699145299146E+01;
//   } else if (x < 0.2928770209555256) {
//     return 5.36891330891336E+02 * x - 2.92588522588527E+01;
//   } else if (x < 0.4061551302245808) {
//     return 6.60915763546766E+02 * x - 6.55827586206742E+01;
//   } else if (x < 0.5) {
//     return 5.64285714285700E+02 * x - 2.63359683794383E+01;
//   } else {
//     return colormap_f(x);
//   }
// }

// vec4 colormap(float x) {
//   float r = clamp(colormap_red(x) / 255.0, 0.0, 1.0);
//   float g = clamp(colormap_green(x) / 255.0, 0.0, 1.0);
//   float b = clamp(colormap_blue(x) / 255.0, 0.0, 1.0);
//   return vec4(r, g, b, 1.0);
// }

vec4 colormap(float x) {
    float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
    float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
    float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

float random (vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

void main() {
  const float t = now - time;
  if ((t < 0.0) || (t > delta)) discard;

  const float begin_mask = smoothstep(0.0, 0.3, arc / char_length);
  const float end_mask = 1.0 - smoothstep(0.95 * delta, delta, t);
  const float decay_mask = exp(-2.0 * t / delta);
  const float speed_value = speed * delta / char_length;
  const float dash_bound = exp(-0.2 * speed_value * speed_value);
  const float speed_mask = 1.0 - dash_bound;
  const float dash_u = mod(15.0 * varc / char_length, 2.0) - 1.0;
  // const float dash_mask = smoothstep(0.0 * dash_bound, dash_bound, abs(dash_u));
  // const float dash_mask = smoothstep(0.0, 1.0, abs(mod(10.0 * varc / char_length, 2.0) - 1.0));

  float weight = begin_mask * end_mask * decay_mask * speed_mask;

  // const float width_mask = (1.0 - smoothstep(0.95 * weight, weight, abs(v)));
  // weight *= width_mask;

  // frag_color = vec4(vec3(0.5), weight);
  if (abs(v) >= 1.0)
    frag_color = vec4(vec3(1.0), weight);
  else
    frag_color = vec4(vec3(colormap(t / delta / 0.7 + 0.3 * random(vec2(float(stroke)/7.0, float(stroke)/13.0)))), weight) ;

  // frag_color = mix(vec4(1.0), colormap(0.1 + t / delta / 0.9), weight);
}
