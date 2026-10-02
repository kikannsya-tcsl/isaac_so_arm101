import carb

settings = carb.settings.get_settings()

print("rendermode:", settings.get("/rtx/rendermode"))

print("RT fractionalCutoutOpacity", settings.get("/rtx/raytracing/fractionalCutoutOpacity"))

print("Translucency Enabled :", settings.get("/rtx/translucency/enabled"))
