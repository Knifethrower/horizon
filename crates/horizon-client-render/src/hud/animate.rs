//! The GUI atlas's animated sprites, frame by frame (vanilla's or a Polytone `mode`'s).

impl super::HudRenderer {
    /// Shows each animated sprite's frame: its mcmeta's, or what its Polytone `mode` picks.
    pub(crate) fn animate(
        &mut self,
        uploads: &mut crate::gfx::upload::Uploads,
        encoder: &mut wgpu::CommandEncoder,
        tick: u64,
        animator: &crate::gfx::animate::SpriteAnimator,
    ) {
        let inputs = animator.mode_inputs();
        let inputs = Some(&inputs);
        let (width, height) = self.atlas.size();
        for (rect, animation, previous) in &mut self.sprite_animations {
            // `SpriteContents.AnimationState`: the frame from the mcmeta's
            // list, blended into the next one by the sub-frame's thousandths
            // when the sprite interpolates; or what a Polytone `mode` picks.
            let blend = animation.blend_with(tick, inputs);
            let state = (blend.frame, blend.next, blend.millis);
            if state == *previous {
                continue;
            }
            let (w, h) = rect.texels((width, height));
            let pixels = if blend.millis == 0 {
                animation.resized_pixels(blend.frame, w, h)
            } else {
                let current = animation.resized_pixels(blend.frame, w, h);
                let next = animation.resized_pixels(blend.next, w, h);
                let millis = blend.millis;
                current
                    .iter()
                    .zip(next)
                    .map(|(&a, b)| {
                        ((u32::from(a) * (1000 - millis) + u32::from(b) * millis + 500) / 1000)
                            as u8
                    })
                    .collect()
            };
            self.atlas.write(
                uploads,
                encoder,
                (rect.u0 * width as f32).round() as u32,
                (rect.v0 * height as f32).round() as u32,
                w,
                h,
                &pixels,
            );
            *previous = state;
        }
    }
}
