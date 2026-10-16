#version 330
in vec2 v_uv;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_aux;
layout(location = 2) out vec4 o_dep;

const int NJ = 18;
uniform vec3 uJ[144];
uniform mat3 uF[24];
uniform vec4 uP[8];      // scale, female, skin+0.1*hair, kit
uniform int uN;
uniform vec3 uCamPos;
uniform mat3 uCamRot;
uniform float uTanHalf;
uniform vec2 uRes;
uniform vec3 uSun, uSunCol, uSky, uGnd, uRimCol;
uniform float uRim, uGround, uExposure;

// materials
const float SKIN = 1., SINGLET = 2., SHORTS = 3., SHOE = 4., SOLE = 5., HAIR = 6., SOCK = 7., EYE = 8., CAP = 9.;

float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / max(k1, 1e-6);
}
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;
    vec3 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    vec3 xv = pa * l2 - ba * y;
    float x2 = dot(xv, xv);
    float y2 = y * y * l2;
    float z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}
vec2 U(vec2 a, vec2 b) { return a.x < b.x ? a : b; }
vec2 SU(vec2 a, vec2 b, float k) {
    float h = max(k - abs(a.x - b.x), 0.0) / k;
    float d = min(a.x, b.x) - h * h * k * 0.25;
    return vec2(d, a.x < b.x ? a.y : b.y);
}
float smin1(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
float smax(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return max(a, b) + h * h * k * 0.25;
}

vec2 limbArm(vec3 p, vec3 sh, vec3 el, vec3 wr, vec3 side, float s) {
    vec3 ud = normalize(el - sh);
    vec3 fd = normalize(wr - el);
    vec3 fr = fd - dot(fd, ud) * ud;
    fr = length(fr) > 1e-3 ? normalize(fr) : side;
    vec2 r = vec2(sdEllipsoid(p - sh - side * 0.012 * s - ud * 0.02 * s, vec3(0.056, 0.068, 0.06) * s), SKIN);
    r = SU(r, vec2(sdRoundCone(p, sh, el, 0.047 * s, 0.036 * s), SKIN), 0.03 * s);
    r = SU(r, vec2(sdRoundCone(p, sh + ud * 0.09 * s + fr * 0.016 * s, sh + ud * 0.23 * s + fr * 0.012 * s, 0.041 * s, 0.03 * s), SKIN), 0.035 * s);
    r = SU(r, vec2(sdRoundCone(p, sh + ud * 0.06 * s - fr * 0.018 * s, sh + ud * 0.24 * s - fr * 0.014 * s, 0.043 * s, 0.03 * s), SKIN), 0.035 * s);
    r = SU(r, vec2(sdRoundCone(p, el, wr, 0.038 * s, 0.025 * s), SKIN), 0.02 * s);
    r = SU(r, vec2(sdRoundCone(p, el + fd * 0.025 * s - fr * 0.006 * s, el + fd * 0.13 * s, 0.041 * s, 0.029 * s), SKIN), 0.03 * s);
    // loose fist: palm flat against the swing plane, thumb on top
    vec3 th = -ud - dot(-ud, fd) * fd;
    th = length(th) > 1e-3 ? normalize(th) : fr;
    vec3 tb = cross(fd, th);
    vec3 hq = p - wr - fd * 0.048 * s;
    vec3 hl = vec3(dot(hq, tb), dot(hq, th), dot(hq, fd));
    float fist = sdEllipsoid(hl, vec3(0.024, 0.04, 0.046) * s);
    fist = smin1(fist, sdEllipsoid(hl - vec3(0.006, -0.004, 0.034) * s, vec3(0.024, 0.037, 0.02) * s), 0.02 * s);
    r = SU(r, vec2(fist, SKIN), 0.014 * s);
    r = SU(r, vec2(sdRoundCone(p, wr + fd * 0.025 * s + th * 0.03 * s, wr + fd * 0.07 * s + th * 0.034 * s + tb * 0.012 * s, 0.012 * s, 0.01 * s), SKIN), 0.01 * s);
    return r;
}

vec2 limbLeg(vec3 p, vec3 hip, vec3 kn, vec3 an, vec3 toe, mat3 P, float s) {
    vec3 td = normalize(kn - hip);
    vec3 pz = P[2];
    vec3 fw = normalize(pz - dot(pz, td) * td);
    float tl = length(kn - hip);
    vec2 r = vec2(sdRoundCone(p, hip, kn, 0.088 * s, 0.054 * s), SKIN);
    r = SU(r, vec2(sdRoundCone(p, hip + td * 0.1 * s + fw * 0.022 * s, kn - td * 0.06 * s + fw * 0.02 * s, 0.075 * s, 0.05 * s), SKIN), 0.05 * s);
    r = SU(r, vec2(sdRoundCone(p, hip + td * 0.08 * s - fw * 0.03 * s, kn - td * 0.1 * s - fw * 0.022 * s, 0.072 * s, 0.045 * s), SKIN), 0.05 * s);
    r = SU(r, vec2(sdRoundCone(p, hip + td * 0.2 * s + cross(td, fw) * 0.035 * s, kn - td * 0.08 * s + cross(td, fw) * 0.02 * s, 0.05 * s, 0.035 * s), SKIN), 0.04 * s);
    r = SU(r, vec2(sdSphere(p - kn - fw * 0.008 * s, 0.05 * s), SKIN), 0.03 * s);
    float f = dot(p - hip, td) / tl;
    if (f < 0.42) { r.x -= 0.009 * s * smoothstep(0.42, 0.36, f); r.y = SHORTS; }
    vec3 sd = normalize(an - kn);
    vec3 bk = -normalize(pz - dot(pz, sd) * sd);
    r = SU(r, vec2(sdRoundCone(p, kn, an, 0.048 * s, 0.03 * s), SKIN), 0.02 * s);
    r = SU(r, vec2(sdRoundCone(p, kn + sd * 0.07 * s + bk * 0.028 * s, kn + sd * 0.25 * s + bk * 0.01 * s, 0.054 * s, 0.033 * s), SKIN), 0.05 * s);
    r = SU(r, vec2(sdRoundCone(p, an, an - sd * 0.075 * s, 0.035 * s, 0.034 * s), SOCK), 0.01 * s);
    // shoe: heel counter + forefoot + flat sole, in foot space
    vec3 fd = normalize(toe - an);
    vec3 fx = normalize(P[0]);
    vec3 fu = normalize(cross(fd, fx));
    vec3 lp = vec3(dot(p - an, fx), dot(p - an, fu), dot(p - an, fd)) / s;
    float up = sdEllipsoid(lp - vec3(0.0, -0.03, -0.02), vec3(0.042, 0.05, 0.075));
    up = smin1(up, sdEllipsoid(lp - vec3(0.0, -0.045, 0.1), vec3(0.046, 0.03, 0.095)), 0.04);
    float sole = sdEllipsoid(lp - vec3(0.0, -0.072, -0.035), vec3(0.04, 0.017, 0.075));
    sole = smin1(sole, sdEllipsoid(lp - vec3(0.0, -0.07, 0.1), vec3(0.047, 0.015, 0.1)), 0.04);
    sole = max(sole, -(lp.y + 0.086));
    vec2 shoe = SU(vec2(up * s, SHOE), vec2(sole * s, SOLE), 0.008 * s);
    shoe.y = lp.y < -0.066 ? SOLE : SHOE;
    r = SU(r, shoe, 0.012 * s);
    return r;
}

vec2 mapRunner(vec3 p, int i) {
    int b = i * NJ;
    vec3 J0 = uJ[b], J1 = uJ[b + 1], J2 = uJ[b + 2], J3 = uJ[b + 3];
    float s = uP[i].x;
    float bnd = sdCapsule(p, J0 - vec3(0.0, 1.0 * s, 0.0), J3, 0.72 * s);
    if (bnd > 0.16) return vec2(bnd, 0.0);
    mat3 P = uF[i * 3], C = uF[i * 3 + 1], Hm = uF[i * 3 + 2];
    float fem = uP[i].y;
    float base = float(i) * 10.0;

    // torso
    vec3 q = transpose(C) * (p - J1);
    vec2 r = vec2(sdEllipsoid(q - vec3(0.0, -0.16, 0.0) * s, vec3(0.148 - 0.012 * fem, 0.17, 0.108) * s), SINGLET);
    vec3 m = mix(J0, J1, 0.45);
    r = SU(r, vec2(sdEllipsoid(transpose(C) * (p - m), vec3(0.142 - 0.01 * fem, 0.15, 0.098) * s), SINGLET), 0.06 * s);
    r = SU(r, vec2(sdEllipsoid(transpose(C) * (p - mix(J0, J1, 0.72)), vec3(0.146 - 0.012 * fem, 0.12, 0.1) * s), SINGLET), 0.06 * s);
    if (fem > 0.5) {
        r = SU(r, vec2(sdEllipsoid(q - vec3(0.052, -0.135, 0.085) * s, vec3(0.055, 0.05, 0.04) * s), SINGLET), 0.03 * s);
        r = SU(r, vec2(sdEllipsoid(q - vec3(-0.052, -0.135, 0.085) * s, vec3(0.055, 0.05, 0.04) * s), SINGLET), 0.03 * s);
    }
    if (fem < 0.5) {
        r = SU(r, vec2(sdEllipsoid(q - vec3(0.066, -0.095, 0.072) * s, vec3(0.074, 0.05, 0.042) * s), SINGLET), 0.03 * s);
        r = SU(r, vec2(sdEllipsoid(q - vec3(-0.066, -0.095, 0.072) * s, vec3(0.074, 0.05, 0.042) * s), SINGLET), 0.03 * s);
    }
    float lw = 1.0 - 0.25 * fem;
    r = SU(r, vec2(sdEllipsoid(q - vec3(0.097 * lw, -0.16, -0.02) * s, vec3(0.055 * lw, 0.12, 0.075) * s), SINGLET), 0.05 * s);
    r = SU(r, vec2(sdEllipsoid(q - vec3(-0.097 * lw, -0.16, -0.02) * s, vec3(0.055 * lw, 0.12, 0.075) * s), SINGLET), 0.05 * s);
    r = SU(r, vec2(sdEllipsoid(q - vec3(0.068, -0.085, -0.075) * s, vec3(0.06, 0.07, 0.035) * s), SINGLET), 0.04 * s);
    r = SU(r, vec2(sdEllipsoid(q - vec3(-0.068, -0.085, -0.075) * s, vec3(0.06, 0.07, 0.035) * s), SINGLET), 0.04 * s);
    vec3 qp = transpose(P) * (p - J0);
    r = SU(r, vec2(sdEllipsoid(qp - vec3(0.0, -0.025, -0.008) * s, vec3(0.158 + 0.014 * fem, 0.11, 0.105) * s), SHORTS), 0.05 * s);
    r = SU(r, vec2(sdSphere(qp - vec3(0.062, -0.065, -0.05) * s, 0.074 * s), SHORTS), 0.04 * s);
    r = SU(r, vec2(sdSphere(qp - vec3(-0.062, -0.065, -0.05) * s, 0.074 * s), SHORTS), 0.04 * s);
    // shoulders / trapezius
    vec3 shL = uJ[b + 4], shR = uJ[b + 7];
    r = SU(r, vec2(sdRoundCone(p, J2 - C[1] * 0.01 * s, shL, 0.058 * s, 0.045 * s), SINGLET), 0.06 * s);
    r = SU(r, vec2(sdRoundCone(p, J2 - C[1] * 0.01 * s, shR, 0.058 * s, 0.045 * s), SINGLET), 0.06 * s);
    // neck + head
    vec3 hu = Hm[1];
    r = SU(r, vec2(sdRoundCone(p, J2, J3 - hu * 0.07 * s + Hm[2] * 0.005 * s, 0.058 * s - 0.006 * s * fem, 0.05 * s - 0.005 * s * fem), SKIN), 0.035 * s);
    vec3 qh = transpose(Hm) * (p - J3);
    vec2 h = vec2(sdEllipsoid(qh - vec3(0.0, 0.01, -0.006) * s, vec3(0.076, 0.097, 0.094) * s), SKIN);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.0, -0.055, 0.03) * s, vec3(0.057, 0.05, 0.06) * s), SKIN), 0.035 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.0, 0.034, 0.068) * s, vec3(0.064, 0.02, 0.03) * s), SKIN), 0.02 * s);
    h = SU(h, vec2(sdRoundCone(qh, vec3(0.0, 0.008, 0.086) * s, vec3(0.0, -0.026, 0.1) * s, 0.009 * s, 0.015 * s), SKIN), 0.012 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.077, -0.002, -0.006) * s, vec3(0.012, 0.027, 0.019) * s), SKIN), 0.008 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(-0.077, -0.002, -0.006) * s, vec3(0.012, 0.027, 0.019) * s), SKIN), 0.008 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.044, -0.008, 0.06) * s, vec3(0.028, 0.02, 0.026) * s), SKIN), 0.02 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(-0.044, -0.008, 0.06) * s, vec3(0.028, 0.02, 0.026) * s), SKIN), 0.02 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.0, -0.092, 0.056) * s, vec3(0.03, 0.022, 0.024) * s), SKIN), 0.02 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.0, -0.058, 0.084) * s, vec3(0.022, 0.009, 0.012) * s), SKIN), 0.01 * s);
    // eye sockets with dark eyes
    float sock = min(sdSphere(qh - vec3(0.031, 0.014, 0.092) * s, 0.016 * s), sdSphere(qh - vec3(-0.031, 0.014, 0.092) * s, 0.016 * s));
    h.x = smax(h.x, -sock, 0.012 * s);
    h = SU(h, vec2(sdEllipsoid(qh - vec3(0.0, 0.018, -0.032) * s, vec3(0.074, 0.086, 0.08) * s), SKIN), 0.03 * s);
    // hair
    float hr = sdEllipsoid(qh - vec3(0.0, 0.028, -0.014) * s, vec3(0.083, 0.094, 0.1) * s);
    hr = max(hr, dot(qh, normalize(vec3(0.0, -0.62, 0.78))) - 0.006 * s);
    if (fem > 0.5) {
        float sw = sin(uP[i].z * 0.0 + J0.y * 40.0) * 0.03;
        hr = min(hr, sdRoundCone(qh, vec3(0.0, 0.05, -0.09) * s, vec3(sw, -0.13 + 2.0 * (J0.y - 0.9), -0.2) * s, 0.032 * s, 0.013 * s));
    }
    h = SU(h, vec2(hr, HAIR), 0.012 * s);
    float acc = floor(uP[i].w * 0.5 + 0.01);
    if (acc > 0.5 && acc < 1.5) {
        float cp = sdEllipsoid(qh - vec3(0.0, 0.034, -0.012) * s, vec3(0.088, 0.092, 0.104) * s);
        cp = max(cp, -(qh.y - 0.022 * s + 0.25 * qh.z));
        float br = sdEllipsoid(qh - vec3(0.0, 0.03, 0.105) * s, vec3(0.08, 0.008, 0.07) * s);
        br = max(br, -(qh.z - 0.06 * s));
        h = U(h, vec2(min(cp, br), CAP));
    }
    if (acc > 0.5 && acc < 2.5) {
        vec3 g = qh - vec3(0.0, 0.014, 0.087) * s;
        float gl = sdEllipsoid(g, vec3(0.074, 0.019, 0.03) * s);
        gl = max(gl, -sdEllipsoid(g + vec3(0.0, 0.0, 0.03) * s, vec3(0.07, 0.06, 0.04) * s));
        h = U(h, vec2(gl, EYE));
    }
    if (acc > 2.5) {
        float hb = sdEllipsoid(qh - vec3(0.0, 0.045, -0.006) * s, vec3(0.081, 0.03, 0.098) * s);
        hb = max(hb, abs(qh.y - 0.045 * s) - 0.014 * s);
        h = U(h, vec2(hb, CAP));
    }
    r = SU(r, h, 0.02 * s);
    // limbs
    r = SU(r, limbArm(p, shL, uJ[b + 5], uJ[b + 6], -C[0], s), 0.03 * s);
    r = SU(r, limbArm(p, shR, uJ[b + 8], uJ[b + 9], C[0], s), 0.03 * s);
    r = SU(r, limbLeg(p, uJ[b + 10], uJ[b + 11], uJ[b + 12], uJ[b + 13], P, s), 0.035 * s);
    r = SU(r, limbLeg(p, uJ[b + 14], uJ[b + 15], uJ[b + 16], uJ[b + 17], P, s), 0.035 * s);
    r.y += base;
    return r;
}

vec2 map(vec3 p) {
    vec2 r = vec2(1e5, -1.0);
    for (int i = 0; i < uN; i++) r = U(r, mapRunner(p, i));
    return r;
}

vec3 calcNormal(vec3 p, float t) {
    float e = 0.0006 + 0.0002 * t;
    vec2 k = vec2(1.0, -1.0);
    return normalize(k.xyy * map(p + k.xyy * e).x + k.yyx * map(p + k.yyx * e).x +
                     k.yxy * map(p + k.yxy * e).x + k.xxx * map(p + k.xxx * e).x);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float softShadow(vec3 ro, vec3 rd, float maxt) {
    float res = 1.0;
    float t = 0.03 + 0.03 * hash12(gl_FragCoord.xy);
    for (int i = 0; i < 56 && t < maxt; i++) {
        float h = map(ro + rd * t).x;
        res = min(res, 7.0 * h / t);
        if (res < 0.003) break;
        t += clamp(h * 0.8, 0.015, 0.15) * (0.8 + 0.4 * hash12(gl_FragCoord.xy + float(i)));
    }
    res = clamp(res, 0.0, 1.0);
    return res * res * (3.0 - 2.0 * res);
}

float calcAO(vec3 p, vec3 n) {
    float occ = 0.0, sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.012 + 0.05 * float(i);
        occ += (h - map(p + n * h).x) * sca;
        sca *= 0.8;
    }
    return clamp(1.0 - 2.2 * occ, 0.0, 1.0);
}

vec3 albedo(vec3 p, float mat, out float spec, out float wrap) {
    int i = int(floor(mat / 10.0 + 0.001));
    float m = mat - float(i) * 10.0;
    float s = uP[i].x;
    float kit = mod(uP[i].w, 2.0);
    float skin = floor(uP[i].z);
    spec = 0.04;
    wrap = 0.0;
    vec3 skinCol = skin < 0.5 ? vec3(0.6, 0.42, 0.33) : (skin < 1.5 ? vec3(0.5, 0.34, 0.25) : vec3(0.68, 0.5, 0.4));
    vec3 red = vec3(0.70, 0.035, 0.05);
    if (m < SKIN + 0.5) { spec = 0.22; wrap = 0.4; float nz = sin(p.x * 61.0) * sin(p.y * 47.0) * sin(p.z * 53.0); return skinCol * (0.96 + 0.04 * nz); }
    if (m < SINGLET + 0.5) {
        vec3 q = transpose(uF[i * 3 + 1]) * (p - uJ[i * NJ + 1]) / s;
        vec3 main = kit < 0.5 ? red : vec3(0.92, 0.9, 0.86);
        vec3 alt = kit < 0.5 ? vec3(0.95, 0.93, 0.9) : red;
        float band = abs(q.x * 0.75 + q.y + 0.2);
        vec3 c = band < 0.032 ? alt : main;
        if (q.z > 0.05 && abs(q.x) < 0.075 && q.y < -0.19 && q.y > -0.31) {
            c = vec3(0.95);
            float dig = step(0.5, fract((q.x + 0.075) * 40.0)) * step(-0.285, q.y) * step(q.y, -0.215) * step(abs(q.x), 0.055);
            if (q.y > -0.215 && q.y < -0.195) c = red;
            c = mix(c, vec3(0.08), dig * 0.85);
        }
        wrap = 0.2;
        return c;
    }
    if (m < SHORTS + 0.5) { spec = 0.08; return vec3(0.025, 0.025, 0.03); }
    if (m < SHOE + 0.5) { spec = 0.2; return kit < 0.5 ? vec3(0.06, 0.06, 0.07) : vec3(0.75, 0.06, 0.07); }
    if (m < SOLE + 0.5) return vec3(0.88, 0.86, 0.82);
    if (m < HAIR + 0.5) { spec = 0.25; return vec3(0.025, 0.02, 0.018); }
    if (m < SOCK + 0.5) return vec3(0.9, 0.9, 0.88);
    if (m < EYE + 0.5) { spec = 0.95; return vec3(0.015, 0.02, 0.03); }
    spec = 0.1; wrap = 0.2;
    return kit < 0.5 ? vec3(0.9, 0.89, 0.86) : vec3(0.68, 0.04, 0.05);
}

vec3 shade(vec3 p, vec3 n, vec3 rd, float mat, float t) {
    float spec, wrap;
    vec3 alb = albedo(p, mat, spec, wrap);
    float occ = calcAO(p, n);
    float sh = softShadow(p + n * 0.02, uSun, 3.0);
    float nl = dot(n, uSun);
    float dif = clamp((nl + wrap) / (1.0 + wrap), 0.0, 1.0);
    vec3 amb = mix(uGnd, uSky, 0.5 + 0.5 * n.y) * occ;
    vec3 lin = uSunCol * dif * sh * 2.0 + amb * 0.95;
    // warm subsurface bleed in the terminator for skin
    if (wrap > 0.3) lin += uSunCol * vec3(0.5, 0.12, 0.06) * smoothstep(-0.25, 0.15, nl) * smoothstep(0.45, 0.0, nl) * 0.6;
    vec3 col = alb * lin;
    vec3 hv = normalize(uSun - rd);
    col += uSunCol * pow(clamp(dot(n, hv), 0.0, 1.0), mix(14.0, 60.0, spec)) * spec * sh * 1.4;
    float fr = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0);
    float back = clamp(dot(rd, uSun) * 0.5 + 0.5, 0.0, 1.0);
    float rim = fr * uRim * (0.25 + 1.6 * back * back) * clamp(nl + 0.55, 0.0, 1.0) * mix(0.5, 1.0, occ);
    col += uRimCol * rim * mix(vec3(1.0), alb * 2.0 + 0.35, 0.35);
    return col * uExposure;
}

void main() {
    vec2 uv = v_uv;
    float aspect = uRes.x / uRes.y;
    vec3 ro = uCamPos;
    vec3 rd = normalize(uCamRot * vec3(uv.x * uTanHalf * aspect, uv.y * uTanHalf, 1.0));
    float px = 2.0 * uTanHalf / uRes.y;
    float tg = (uGround > 0.5 && rd.y < -1e-4) ? -ro.y / rd.y : 1e9;
    float tmax = min(tg, 80.0);
    float t = 0.05;
    vec2 hit = vec2(-1.0);
    float bestR = 1e9, bestT = 0.0, bestM = -1.0;
    for (int i = 0; i < 160; i++) {
        vec3 p = ro + rd * t;
        vec2 h = map(p);
        float rr = h.x / (t * px);
        if (h.y > 0.5 && rr < bestR) { bestR = rr; bestT = t; bestM = h.y; }
        if (h.x < 0.25 * t * px) { hit = vec2(t, h.y); break; }
        t += h.x * 0.9;
        if (t > tmax) break;
    }
    vec4 col = vec4(0.0);
    if (hit.y > 0.5) {
        vec3 p = ro + rd * hit.x;
        vec3 n = calcNormal(p, hit.x);
        col = vec4(shade(p, n, rd, hit.y, hit.x), 1.0);
    } else if (bestR < 1.0 && bestM > 0.5) {
        float a = clamp(1.0 - bestR, 0.0, 1.0);
        vec3 p = ro + rd * bestT;
        vec3 n = calcNormal(p, bestT);
        col = vec4(shade(p, n, rd, bestM, bestT) * a, a);
    }
    vec4 aux = vec4(0.0, 1.0, 0.0, 0.0);
    if (tg < 1e8 && (hit.y < 0.5 || hit.x > tg)) {
        vec3 pg = ro + rd * tg;
        float sh = softShadow(pg + vec3(0.0, 0.002, 0.0), uSun, 4.0);
        float d = map(pg).x;
        float ao = clamp(0.35 + 0.65 * d / 0.12, 0.0, 1.0);
        aux = vec4(1.0 - sh, ao, tg, 1.0);
    }
    o_col = col;
    o_aux = aux;
    o_dep = vec4(hit.y > 0.5 ? hit.x : (bestR < 1.0 && bestM > 0.5 ? bestT : 1e4), 0.0, 0.0, 1.0);
}
