import carb

settings = carb.settings.get_settings()
settings.set("/rtx/translucency/enabled", True)
settings.set("/rtx/translucency/maxRefractionBounces", 6)
settings.set("/rtx/translucency/reflectAtAllBounce", True)