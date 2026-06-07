# Build Agent - Strict Rules (MUST FOLLOW 100%)

## 1. Basic Build Rules
- ONLY use the build guide: **docs/how-to-build-ggmlhexagon.md**
- DO NOT read, parse, or use: **Makefile, CMakeLists.txt, cmake, make, ninja**
- DO NOT infer any build steps automatically
- DO NOT use any top-level build system files
- ONLY run commands that are explicitly written in docs/how-to-build-ggmlhexagon.md
- CHECK whether the project is located on a CIFS file system and modify ./scripts/build-run-android.sh accordingly

## 2. Script Execution Rules
- ONLY execute the official build script:
  **./scripts/build-run-android.sh build**
- NEVER run the script with any other arguments or modes

## 3. Function Blacklist (FORBIDDEN)
- In **./scripts/build-run-android.sh**, you are **STRICTLY FORBIDDEN** to:
  - Call, invoke, or execute the function: **check_android_phone()**
  - Call, invoke, or execute the function: **check_prebuilt_models()**
  - Read, analyze, or expand any of the functions above
  - Add any flags or parameters to enable these functions
- IGNORE these functions completely. Pretend they do NOT exist.
- ONLY run the main build flow defined in the guide.

## 4. Final Constraint
If you are unsure → **STOP**.
Do NOT use make/cmake.
Do NOT run forbidden functions.
ONLY follow: docs/how-to-build-ggmlhexagon.md
