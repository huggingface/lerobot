"use client";

import { useWizard } from "./wizard-provider";
import { WizardSidebar } from "./wizard-sidebar";
import { WizardTopbar } from "./wizard-topbar";
import { RobotTypeStep } from "./steps/robot-type-step";
import { PortsStep } from "./steps/ports-step";
import { CamerasStep } from "./steps/cameras-step";
import { CalibrationStep } from "./steps/calibration-step";
import { TeleoperateStep } from "./steps/teleoperate-step";
import { RecordStep } from "./steps/record-step";

const STEP_COMPONENTS = [
  RobotTypeStep,
  PortsStep,
  CamerasStep,
  CalibrationStep,
  TeleoperateStep,
  RecordStep,
];

export function WizardLayout() {
  const { state } = useWizard();
  const StepComponent = STEP_COMPONENTS[state.currentStep];

  return (
    <div className="flex min-h-screen bg-muted">
      <WizardSidebar />
      <div className="flex-1 pl-60">
        <WizardTopbar />
        <main className="flex justify-center px-8 py-12">
          <div className="w-full max-w-2xl">
            <StepComponent />
          </div>
        </main>
      </div>
    </div>
  );
}
