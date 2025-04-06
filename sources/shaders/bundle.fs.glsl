#version 460 core

uniform float now = 0.0;
// uniform float delta = 0.6;
uniform float delta = 0.5;
uniform float char_length = 1.0;
uniform float line_width;

in float time;
in float arc;
in float v;
in float varc;
in float sarc;
in float speed;
flat in uint stroke;

layout (location = 0) out vec4 frag_color;

float colormap_red(float x) {
    return (1.0 + 1.0 / 63.0) * x - 1.0 / 63.0;
}

float colormap_green(float x) {
    return -(1.0 + 1.0 / 63.0) * x + (1.0 + 1.0 / 63.0);
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = 1.0;
    return vec4(r, g, b, 1.0);
}

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

// vec4 colormap(float x) {
//     float r = clamp(8.0 / 3.0 * x, 0.0, 1.0);
//     float g = clamp(8.0 / 3.0 * x - 1.0, 0.0, 1.0);
//     float b = clamp(4.0 * x - 3.0, 0.0, 1.0);
//     return vec4(r, g, b, 1.0);
// }

// float colormap_f1(float x) {
//     return -510.0 * x + 255.0;
// }

// float colormap_f2(float x) {
//     return (-1891.7 * x + 217.46) * x + 255.0;
// }

// float colormap_f3(float x) {
//     return 9.26643676359015e1 * sin((x - 4.83450094847127e-1) * 9.93) + 1.35940451627965e2;
// }

// float colormap_f4(float x) {
//     return -510.0 * x + 510.0;
// }

// float colormap_f5(float x) {
//     float xx = x - 197169.0 / 251000.0;
//     return (2510.0 * xx - 538.31) * xx;
// }

// float colormap_red(float x) {
//     if (x < 0.0) {
//         return 1.0;
//     } else if (x < 10873.0 / 94585.0) {
//         float xx = colormap_f2(x);
//         if (xx > 255.0) {
//             return (510.0 - xx) / 255.0;
//         } else {
//             return xx / 255.0;
//         }
//     } else if (x < 0.5) {
//         return 1.0;
//     } else if (x < 146169.0 / 251000.0) {
//         return colormap_f4(x) / 255.0;
//     } else if (x < 197169.0 / 251000.0) {
//         return colormap_f5(x) / 255.0;
//     } else {
//         return 0.0;
//     }
// }

// float colormap_green(float x) {
//     if (x < 10873.0 / 94585.0) {
//         return 1.0;
//     } else if (x < 36373.0 / 94585.0) {
//         return colormap_f2(x) / 255.0;
//     } else if (x < 0.5) {
//         return colormap_f1(x) / 255.0;
//     } else if (x < 197169.0 / 251000.0) {
//         return 0.0;
//     } else if (x <= 1.0) {
//         return abs(colormap_f5(x)) / 255.0;
//     } else {
//         return 0.0;
//     }
// }

// float colormap_blue(float x) {
//     if (x < 0.0) {
//         return 0.0;
//     } else if (x < 36373.0 / 94585.0) {
//         return colormap_f1(x) / 255.0;
//     } else if (x < 146169.0 / 251000.0) {
//         return colormap_f3(x) / 255.0;
//     } else if (x <= 1.0) {
//         return colormap_f4(x) / 255.0;
//     } else {
//         return 0.0;
//     }
// }

// vec4 colormap(float x) {
//     return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
// }

// float colormap_f1(int formula, float x) {
//     float DEG2RAD = 3.1415926535897932384 / 180.0;
//     if (formula < 0) {
//         x = 1.0 - x;
//         formula = -formula;
//     }
//     float d = 0.0;
//     if (formula == 0) {
//         return 0.0;
//     } else if (formula == 1) {
//         return 0.5;
//     } else if (formula == 2) {
//         return 1.0;
//     } else if (formula == 3) {
//         d = x;
//     } else if (formula == 4) {
//         d = x * x;
//     } else if (formula == 5) {
//         d = x * x * x;
//     } else if (formula == 6) {
//         d = x * x * x * x;
//     } else if (formula == 7) {
//         d = sqrt(x);
//     } else if (formula == 8) {
//         d = sqrt(sqrt(x));
//     } else if (formula == 9) {
//         d = sin(90.0 * x * DEG2RAD);
//     } else if (formula == 10) {
//         d = cos(90.0 * x * DEG2RAD);
//     } else if (formula == 11) {
//         d = abs(x - 0.5);
//     } else if (formula == 12) {
//         d = (2.0 * x - 1.0) * (2.0 * x - 1.0);
//     } else if (formula == 13) {
//         d = sin(180.0 * x * DEG2RAD);
//     } else if (formula == 14) {
//         d = abs(cos(180.0 * x * DEG2RAD));
//     } else if (formula == 15) {
//         d = sin(360.0 * x * DEG2RAD);
//     } else if (formula == 16) {
//         d = cos(360.0 * x * DEG2RAD);
//     } else if (formula == 17) {
//         d = abs(sin(360.0 * x * DEG2RAD));
//     } else if (formula == 18) {
//         d = abs(cos(360.0e0 * x * DEG2RAD));
//     } else if (formula == 19) {
//         d = abs(sin(720.0e0 * x * DEG2RAD));
//     } else if (formula == 20) {
//         d = abs(cos(720.0e0 * x * DEG2RAD));
//     } else if (formula == 21) {
//         d = 3.0e0 * x;
//     } else if (formula == 22) {
//         d = 3.0e0 * x - 1.0e0;
//     } else if (formula == 23) {
//         d = 3.0e0 * x - 2.0e0;
//     } else if (formula == 24) {
//         d = abs(3.0e0 * x - 1.0e0);
//     } else if (formula == 25) {
//         d = abs(3.0e0 * x - 2.0e0);
//     } else if (formula == 26) {
//         d = 1.5e0 * x - 0.5e0;
//     } else if (formula == 27) {
//         d = 1.5e0 * x - 1.0e0;
//     } else if (formula == 28) {
//         d = abs(1.5e0 * x - 0.5e0);
//     } else if (formula == 29) {
//         d = abs(1.5e0 * x - 1.0e0);
//     } else if (formula == 30) {
//         if (x <= 0.25e0) {
//             return 0.0;
//         } else if (x >= 0.57) {
//             return 1.0;
//         } else {
//             d = x / 0.32 - 0.78125;
//         }
//     } else if (formula == 31) {
//         if (x <= 0.42) {
//             return 0.0;
//         } else if (x >= 0.92) {
//             return d = 1.0;
//         } else {
//             d = 2.0 * x - 0.84;
//         }
//     } else if (formula == 32) {
//         if (x <= 0.42) {
//             d = x * 4.0;
//         } else {
//             if (x <= 0.92e0) {
//                 d = -2.0 * x + 1.84;
//             } else {
//                 d = x / 0.08 - 11.5;
//             }
//         }
//     } else if (formula == 33) {
//         d = abs(2.0 * x - 0.5);
//     } else if (formula == 34) {
//         d = 2.0 * x;
//     } else if (formula == 35) {
//         d = 2.0 * x - 0.5;
//     } else if (formula == 36) {
//         d = 2.0 * x - 1.0;
//     }
//     if (d <= 0.0) {
//         return 0.0;
//     } else if (d >= 1.0) {
//         return 1.0;
//     } else {
//         return d;
//     }
// }

// vec4 colormap(float x, int red_type, int green_type, int blue_type) {
//     return vec4(colormap_f1(red_type, x), colormap_f1(green_type, x), colormap_f1(blue_type, x), 1.0);
// }

// vec4 colormap(float x) {
//   return colormap(1.0 - x, 30, 31, 32);
// }

// float colormap_red(float x) {
//     if (x < 0.75) {
//         return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
//     } else {
//         return (13.0 + 8.0 / 9.0) / 10.0 * x - (3.0 + 8.0 / 9.0) / 10.0;
//     }
// }

// float colormap_green(float x) {
//     if (x <= 0.375) {
//         return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
//     } else if (x <= 0.75) {
//         return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 100.0;
//     } else {
//         return 8.0 / 9.0 * x + 1.0 / 9.0;
//     }
// }

// float colormap_blue(float x) {
//     if (x <= 0.375) {
//         return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 1000.0;
//     } else {
//         return 8.0 / 9.0 * x + 1.0 / 9.0;
//     }
// }

// vec4 colormap(float x) {
//     float r = clamp(colormap_red(x), 0.0, 1.0);
//     float g = clamp(colormap_green(x), 0.0, 1.0);
//     float b = clamp(colormap_blue(x), 0.0, 1.0);
//     return vec4(r, g, b, 1.0);
// }

// vec4 colormap(float x) {
//   float d = clamp(x, 0.0, 1.0);
//   return vec4(d, d, d, 1.0);
// }

// float colormap_red(float x) {
//     if (x < 0.0) {
//         return 124.0 / 255.0;
//     } else if (x <= 1.0) {
//         return (128.0 * sin(6.25 * (x + 0.5)) + 128.0) / 255.0;
//     } else {
//         return 134.0 / 255.0;
//     }
// }


// float colormap_green(float x) {
//     if (x < 0.0) {
//         return 121.0 / 255.0;
//     } else if (x <= 1.0) {
//         return (63.0 * sin(x * 99.72) + 97.0) / 255.0;
//     } else {
//         return 52.0 / 255.0;
//     }
// }

// float colormap_blue(float x) {
//     if (x < 0.0) {
//         return 131.0 / 255.0;
//     } else if (x <= 1.0) {
//         return (128.0 * sin(6.23 * x) + 128.0) / 255.0;
//     } else {
//         return 121.0 / 255.0;
//     }
// }

// vec4 colormap(float x) {
//     return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
// }

// float colormap_red(float x) {
//   if (x < 0.09082479229584027) {
//     return 1.24879652173913E+03 * x + 1.41460000000000E+02;
//   } else if (x < 0.1809653122266933) {
//     return -7.21339920948626E+02 * x + 3.20397233201581E+02;
//   } else if (x < 0.2715720097177793) {
//     return 6.77416996047422E+02 * x + 6.72707509881444E+01;
//   } else if (x < 0.3619607687891861) {
//     return -1.36850782608711E+03 * x + 6.22886666666710E+02;
//   } else if (x < 0.4527609316115322) {
//     return 1.38118774703557E+03 * x - 3.72395256916997E+02;
//   } else if (x < 0.5472860687991931) {
//     return -7.81436521739194E+02 * x + 6.06756521739174E+02;
//   } else if (x < 0.6360981817705944) {
//     return 8.06836521739242E+02 * x - 2.62483188405869E+02;
//   } else if (x < 0.8158623444475089) {
//     return -3.49616157878512E+02 * x + 4.73134258402717E+02;
//   } else if (x < 0.9098023786863947) {
//     return 1.72428853754953E+02 * x + 4.72173913043111E+01;
//   } else {
//     return 5.44142292490101E+02 * x - 2.90968379446626E+02;
//   }
// }

// float colormap_green(float x) {
//   if (x < 0.08778161310534617) {
//     return 4.88563478260870E+02 * x + 2.10796666666667E+02;
//   } else if (x < 0.2697669137324175) {
//     return -6.96835646006769E+02 * x + 3.14852913968545E+02;
//   } else if (x < 0.3622079895714037) {
//     return 5.40799130434797E+02 * x - 1.90200000000068E+01;
//   } else if (x < 0.4519795462045253) {
//     return 3.23774703557373E+01 * x + 1.65134387351785E+02;
//   } else if (x < 0.5466820192751115) {
//     return 4.43064347826088E+02 * x - 2.04876811594176E+01;
//   } else if (x < 0.6368889369442862) {
//     return -1.83472332015826E+02 * x + 3.22028656126484E+02;
//   } else if (x < 0.728402572416003) {
//     return 1.27250988142231E+02 * x + 1.24132411067220E+02;
//   } else if (x < 0.8187333479165154) {
//     return -9.82116600790428E+02 * x + 9.32198616600708E+02;
//   } else if (x < 0.9094607880855196) {
//     return 1.17713438735149E+03 * x - 8.35652173912769E+02;
//   } else {
//     return 2.13339920948864E+01 * x + 2.15502964426857E+02;
//   }
// }

// float colormap_blue(float x) {
//   if (x < 0.09081516507716858) {
//     return -2.27937391304345E+02 * x + 1.99486666666666E+02;
//   } else if (x < 0.1809300436999751) {
//     return 4.33958498023703E+02 * x + 1.39376482213440E+02;
//   } else if (x < 0.2720053156712806) {
//     return -1.14300000000004E+03 * x + 4.24695652173923E+02;
//   } else if (x < 0.3616296568054424) {
//     return 1.08175889328072E+03 * x - 1.80450592885399E+02;
//   } else if (x < 0.4537067088757783) {
//     return -1.22681999999994E+03 * x + 6.54399999999974E+02;
//   } else if (x < 0.5472726179445029) {
//     return 8.30770750988243E+01 * x + 6.00909090909056E+01;
//   } else if (x < 0.6374811920489858) {
//     return 1.36487351778676E+03 * x - 6.41401185770872E+02;
//   } else if (x < 0.7237636846906381) {
//     return -1.27390769230737E+02 * x + 3.09889230769173E+02;
//   } else if (x < 0.8178226469606309) {
//     return -3.01831168831021E+02 * x + 4.36142857142782E+02;
//   } else if (x < 0.9094505664375214) {
//     return 8.47622811970801E+01 * x + 1.19977978543158E+02;
//   } else {
//     return -9.06117391304296E+02 * x + 1.02113405797096E+03;
//   }
// }

// vec4 colormap(float x) {
//   float r = clamp(colormap_red(x) / 255.0, 0.0, 1.0);
//   float g = clamp(colormap_green(x) / 255.0, 0.0, 1.0);
//   float b = clamp(colormap_blue(x) / 255.0, 0.0, 1.0);
//   return vec4(r, g, b, 1.0);
// }

float random (vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

vec2 hash( vec2 p )
{
  p = vec2( dot(p,vec2(127.1,311.7)),
        dot(p,vec2(269.5,183.3)) );
  return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p )
{
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;

  vec2 i = floor( p + (p.x+p.y)*K1 );

    vec2 a = p - i + (i.x+i.y)*K2;
    vec2 o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0); //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    vec2 b = a - o + K2;
  vec2 c = a - 1.0 + 2.0*K2;

    vec3 h = max( 0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 );

  vec3 n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));

    return dot( n, vec3(70.0) );
}
float noise01(vec2 p)
{
    return clamp((noise(p)+.5)*.5, 0.,1.);
}

float udSegment( in vec2 p, in vec2 a, in vec2 b )
{
    vec2 ba = b-a;
    vec2 pa = p-a;
    float h =clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length(pa-h*ba);
}

void main() {
  const float t = now - time;
  if ((t < 0.0) || (t > delta)) discard;

  const float begin_mask = smoothstep(0.0, 0.05, arc / char_length);
  const float end_mask = 1.0 - smoothstep(0.95 * delta, delta, t);
  const float decay_mask = exp(-2.0 * t / delta);
  const float speed_value = speed * delta / char_length;
  // if (speed_value < 1.5) discard;
  const float dash_bound = exp(-0.3 * speed_value);
  const float speed_mask = (1.0 - dash_bound);
  const float dash_u = mod(15.0 * varc / char_length, 2.0) - 1.0;
  // const float dash_mask = smoothstep(0.0 * dash_bound, dash_bound, abs(dash_u));
  const float dash_mask = smoothstep(0.3, 0.4, abs(mod(45.0 * varc / char_length, 2.0) - 1.0));

  float weight = begin_mask * end_mask * decay_mask * speed_mask;
  // weight = smoothstep(0.05, 0.1, weight) * weight;
  // if (weight < 0.05) discard;



  // if (gl_FragCoord.y < 200.0) discard;

  // const float width_mask = (1.0 - smoothstep(0.95 * weight, weight, abs(v)));
  // weight *= width_mask;

  // frag_color = vec4(vec3(0.5), weight);
  // if (abs(v) > 1.2)
  //   frag_color = vec4(vec3(1.0), weight);
  // else

  // vec2 p = 2.*(2.0*gl_FragCoord-iResolution.xy)/iResolution.y;
  // vec2 p = vec2(2.0 * (0.5 * line_width * v - line_width) / sarc, 2.0 * ((sarc - arc) / sarc - 0.5));
  // vec2 v1 = vec2(0.0,1.00);
  // vec2 v2 = vec2(0.0,0.00);
  // vec2 posInLine = smoothstep(v1, v2, p);
  // float _t = posInLine.y;
  // // float _t = ;
  // float _v = udSegment( p, v1, v2 )+0.75;
  // // float _v = abs(v);
  // vec3 col = vec3(1.0);
  // if(_v < 1.) col*=0.;
  // col += vec3(_t);
  // float strokeAlpha = noise01(vec2(p.x, 0.0) * (vec2(29. * pow(weight, 3), 1.) + 0.0 * random(vec2(float(stroke), 1.0))));
  // // + noise01((v2-p) * vec2(24., 1.));
  // col = mix(col, vec3(strokeAlpha) + col, _t * 10.);

  // frag_color = vec4(col, weight);

  // if (abs(v) > 1.0) discard;

  if (abs(v) >= 0.8) {
    const float tmp = smoothstep(0.8, 1.1, abs(v));
    frag_color = mix(vec4(vec3(1.0), weight), vec4(vec3(colormap(t / delta / 0.8 + 0.2)), weight), 1.0 - tmp);
  }
  else {
    const float tmp = smoothstep(0.5, 0.8, abs(v));
    const vec4 begin_color = vec4(vec3(colormap(t / delta / 0.9 + 0.1)), weight);
    const vec4 end_color = vec4(vec3(colormap(t / delta / 0.6 + 0.4)), weight);

    // vec4 end_color = mix(colormap(t / delta), vec4(1.0), (1.0 - abs(cos(5.0 * v))) * (1.0 - exp(-2.0 * t / delta)));

    frag_color = mix(begin_color, end_color, 1.0 - tmp);
    // frag_color = mix(vec4(vec3(colormap(t / delta / 0.8 + 0.2)), weight), vec4(vec3(colormap(t / delta / 1.0 + 0.0 /* * random(vec2(float(stroke)/7.0, float(stroke)/13.0))*/)), weight), 1.0 - tmp);
  }

  // // frag_color = mix(vec4(1.0), colormap(0.1 + t / delta / 0.9), weight);
  frag_color = vec4(vec3(0.0), 1.0);
}
