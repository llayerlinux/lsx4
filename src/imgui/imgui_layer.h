// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace ImGui {

class Layer {
public:
    virtual ~Layer() = default;
    static void AddLayer(Layer* layer);
    static void RemoveLayer(Layer* layer);

    // Dialog/IME/trophy layers are modal by default. Persistent non-modal layers (for example the
    // devtools overlay) opt out so platform frontends know when gamepad input belongs to ImGui.
    virtual bool CapturesGamepadInput() const {
        return true;
    }
    virtual void Draw() = 0;
};

} // namespace ImGui
