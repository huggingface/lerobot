"use client";

import { WizardProvider } from "@/components/wizard/wizard-provider";
import { WizardLayout } from "@/components/wizard/wizard-layout";

export default function Home() {
  return (
    <WizardProvider>
      <WizardLayout />
    </WizardProvider>
  );
}
